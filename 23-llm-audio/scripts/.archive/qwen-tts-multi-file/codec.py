"""
Neural speech codecs (tokenizers) for Qwen3-TTS.

The codec turns a raw waveform into a short sequence of discrete tokens that the
language model can predict, and turns tokens back into audio. Two variants
mirror the two paths in the paper:

  * MultiCodebookCodec (the "12Hz" family)
      Mimi/EnCodec-style: a fully *causal* conv encoder downsamples the
      waveform into frames, a Residual Vector Quantizer (RVQ) turns each frame
      into NQ codebook indices (codebook 0 = semantic, the rest add acoustic
      detail), and a causal conv decoder reconstructs the waveform directly.

  * SingleCodebookCodec (the "25Hz" family)
      A single VQ codebook emphasizing semantic content. It only produces the
      discrete tokens; reconstruction is handed to the Flow-Matching DiT +
      vocoder in `flow_dit.py`. It also exposes a mel-spectrogram target used
      to train that decoder.

Everything runs on toy-scale audio. Frame rate = sample_rate / prod(strides);
we use small strides so a couple of seconds of audio become ~50-100 frames,
which keeps the language model sequences short and training fast on a laptop.
The "12Hz"/"25Hz" names refer to the design family, not the literal rate here.
"""
import torch
import torch.nn as nn
import torch.nn.functional as F

import torchaudio


class CausalConv1d(nn.Module):
    """Conv1d that only sees the past: left-pad by (kernel-1)*dilation."""

    def __init__(self, in_ch, out_ch, kernel_size, stride=1, dilation=1):
        super().__init__()
        self.pad = (kernel_size - 1) * dilation
        self.conv = nn.Conv1d(in_ch, out_ch, kernel_size, stride=stride, dilation=dilation)

    def forward(self, x):
        # x: (batch, channels, time). Pad on the left only, then trim any
        # extra frames the stride may produce on the right.
        x = F.pad(x, (self.pad, 0))
        return self.conv(x)


class CausalConvTranspose1d(nn.Module):
    """Transposed conv for upsampling, trimmed to stay causal."""

    def __init__(self, in_ch, out_ch, kernel_size, stride):
        super().__init__()
        self.stride = stride
        self.kernel_size = kernel_size
        self.conv = nn.ConvTranspose1d(in_ch, out_ch, kernel_size, stride=stride)

    def forward(self, x):
        x = self.conv(x)
        # Remove the trailing samples introduced by the transposed conv so the
        # output length is exactly stride * input_length.
        extra = self.kernel_size - self.stride
        if extra > 0:
            x = x[..., :-extra]
        return x


class ResidualUnit(nn.Module):
    """Small causal residual block used inside the encoder/decoder."""

    def __init__(self, channels, dilation=1):
        super().__init__()
        self.conv1 = CausalConv1d(channels, channels, kernel_size=3, dilation=dilation)
        self.conv2 = CausalConv1d(channels, channels, kernel_size=1)

    def forward(self, x):
        h = F.elu(self.conv1(F.elu(x)))
        h = self.conv2(h)
        return x + h


class ConvEncoder(nn.Module):
    """Waveform (batch, 1, time) -> latent frames (batch, latent_dim, frames)."""

    def __init__(self, latent_dim, base_channels=32, strides=(5, 4, 4, 4)):
        super().__init__()
        layers = [CausalConv1d(1, base_channels, kernel_size=7)]
        ch = base_channels
        for stride in strides:
            out_ch = min(ch * 2, 256)
            layers += [
                ResidualUnit(ch, dilation=1),
                ResidualUnit(ch, dilation=3),
                CausalConv1d(ch, out_ch, kernel_size=2 * stride, stride=stride),
            ]
            ch = out_ch
        layers += [CausalConv1d(ch, latent_dim, kernel_size=3)]
        self.net = nn.Sequential(*layers)

    def forward(self, x):
        return self.net(x)


class ConvDecoder(nn.Module):
    """Latent frames (batch, latent_dim, frames) -> waveform (batch, 1, time)."""

    def __init__(self, latent_dim, base_channels=32, strides=(5, 4, 4, 4)):
        super().__init__()
        # Mirror the encoder: start from the encoder's final channel count.
        ch = min(base_channels * (2 ** len(strides)), 256)
        layers = [CausalConv1d(latent_dim, ch, kernel_size=3)]
        for stride in reversed(strides):
            out_ch = max(ch // 2, base_channels)
            layers += [
                CausalConvTranspose1d(ch, out_ch, kernel_size=2 * stride, stride=stride),
                ResidualUnit(out_ch, dilation=1),
                ResidualUnit(out_ch, dilation=3),
            ]
            ch = out_ch
        layers += [CausalConv1d(ch, 1, kernel_size=7)]
        self.net = nn.Sequential(*layers)

    def forward(self, x):
        return torch.tanh(self.net(x))


class VectorQuantizer(nn.Module):
    """A single VQ codebook with straight-through gradients.

    Given continuous frames, snaps each to its nearest codebook entry, returns
    the indices, the quantized vectors, and the VQ training loss (codebook +
    commitment terms).
    """

    def __init__(self, codebook_size, dim, commitment_weight=0.25):
        super().__init__()
        self.codebook_size = codebook_size
        self.dim = dim
        self.commitment_weight = commitment_weight
        self.codebook = nn.Embedding(codebook_size, dim)
        nn.init.uniform_(self.codebook.weight, -1.0 / codebook_size, 1.0 / codebook_size)

    def forward(self, z):
        """z: (batch, frames, dim) -> (quantized, indices, loss)."""
        flat = z.reshape(-1, self.dim)  # (N, dim)
        # Squared L2 distance to every code, then argmin.
        dist = (
            flat.pow(2).sum(1, keepdim=True)
            - 2 * flat @ self.codebook.weight.t()
            + self.codebook.weight.pow(2).sum(1)
        )
        indices = dist.argmin(dim=1)  # (N,)
        quantized = self.codebook(indices).view_as(z)

        # VQ-VAE losses: pull the codebook toward encoder outputs (codebook
        # loss) and the encoder toward the codebook (commitment loss).
        codebook_loss = F.mse_loss(quantized, z.detach())
        commitment_loss = F.mse_loss(z, quantized.detach())
        loss = codebook_loss + self.commitment_weight * commitment_loss

        # Straight-through: gradients flow to the encoder as if quantization
        # were the identity.
        quantized = z + (quantized - z).detach()
        return quantized, indices.view(z.shape[:-1]), loss

    def decode_indices(self, indices):
        """indices: (batch, frames) -> quantized vectors (batch, frames, dim)."""
        return self.codebook(indices)


class ResidualVQ(nn.Module):
    """Stack of VQ codebooks; each quantizes the residual left by the last.

    Codebook 0 captures the coarse (semantic) structure; later codebooks add
    finer acoustic detail. This is the core idea behind RVQ tokenizers.
    """

    def __init__(self, num_codebooks, codebook_size, dim, commitment_weight=0.25):
        super().__init__()
        self.num_codebooks = num_codebooks
        self.quantizers = nn.ModuleList(
            [VectorQuantizer(codebook_size, dim, commitment_weight) for _ in range(num_codebooks)]
        )

    def forward(self, z):
        """z: (batch, frames, dim) -> (quantized_sum, codes, loss).

        codes: (batch, frames, num_codebooks) integer indices.
        """
        residual = z
        quantized_sum = torch.zeros_like(z)
        codes = []
        total_loss = 0.0
        for vq in self.quantizers:
            quantized, indices, loss = vq(residual)
            residual = residual - quantized
            quantized_sum = quantized_sum + quantized
            codes.append(indices)
            total_loss = total_loss + loss
        codes = torch.stack(codes, dim=-1)  # (batch, frames, num_codebooks)
        return quantized_sum, codes, total_loss / self.num_codebooks

    def decode_codes(self, codes):
        """codes: (batch, frames, num_codebooks) -> quantized_sum (b, f, dim)."""
        quantized_sum = 0.0
        for i, vq in enumerate(self.quantizers):
            quantized_sum = quantized_sum + vq.decode_indices(codes[..., i])
        return quantized_sum


class MultiCodebookCodec(nn.Module):
    """The 12Hz family: causal conv encoder + RVQ + causal conv decoder.

    encode(wav)  -> codes (batch, frames, num_codebooks)
    decode(codes)-> wav
    forward(wav) -> (recon, codes, vq_loss) for training the codec.
    """

    def __init__(
        self,
        num_codebooks=8,
        codebook_size=1024,
        latent_dim=128,
        base_channels=32,
        strides=(8, 5, 4, 4),
        sample_rate=16000,
    ):
        super().__init__()
        self.num_codebooks = num_codebooks
        self.codebook_size = codebook_size
        self.latent_dim = latent_dim
        self.strides = tuple(strides)
        self.sample_rate = sample_rate
        self.hop = int(torch.prod(torch.tensor(strides)).item())  # samples per frame

        self.encoder = ConvEncoder(latent_dim, base_channels, strides)
        self.rvq = ResidualVQ(num_codebooks, codebook_size, latent_dim)
        self.decoder = ConvDecoder(latent_dim, base_channels, strides)

    def encode(self, wav):
        """wav: (batch, samples) or (batch, 1, samples) -> codes."""
        if wav.dim() == 2:
            wav = wav.unsqueeze(1)
        z = self.encoder(wav).transpose(1, 2)  # (batch, frames, latent_dim)
        _, codes, _ = self.rvq(z)
        return codes

    def decode(self, codes):
        """codes: (batch, frames, num_codebooks) -> wav (batch, 1, samples)."""
        z = self.rvq.decode_codes(codes).transpose(1, 2)  # (batch, latent, frames)
        return self.decoder(z)

    def forward(self, wav):
        if wav.dim() == 2:
            wav = wav.unsqueeze(1)
        z = self.encoder(wav).transpose(1, 2)
        quantized, codes, vq_loss = self.rvq(z)
        recon = self.decoder(quantized.transpose(1, 2))
        return recon, codes, vq_loss


class SingleCodebookCodec(nn.Module):
    """The 25Hz family: conv encoder + single semantic VQ codebook.

    Produces one token per frame. Reconstruction is delegated to the
    Flow-Matching DiT + vocoder, so this module only encodes to tokens and
    (for training that decoder) exposes the quantized/continuous latent.
    """

    def __init__(
        self,
        codebook_size=4096,
        latent_dim=128,
        base_channels=32,
        strides=(5, 4, 4, 4),
        sample_rate=16000,
    ):
        super().__init__()
        self.codebook_size = codebook_size
        self.latent_dim = latent_dim
        self.strides = tuple(strides)
        self.sample_rate = sample_rate
        self.hop = int(torch.prod(torch.tensor(strides)).item())

        self.encoder = ConvEncoder(latent_dim, base_channels, strides)
        self.vq = VectorQuantizer(codebook_size, latent_dim)

    def encode(self, wav):
        """wav -> tokens (batch, frames)."""
        if wav.dim() == 2:
            wav = wav.unsqueeze(1)
        z = self.encoder(wav).transpose(1, 2)
        _, indices, _ = self.vq(z)
        return indices

    def encode_with_loss(self, wav):
        """wav -> (tokens, quantized latent, vq_loss); for codec training."""
        if wav.dim() == 2:
            wav = wav.unsqueeze(1)
        z = self.encoder(wav).transpose(1, 2)
        quantized, indices, loss = self.vq(z)
        return indices, quantized, loss

    def token_embeddings(self, tokens):
        """tokens: (batch, frames) -> codebook vectors (batch, frames, dim)."""
        return self.vq.decode_indices(tokens)


class MelSpectrogram(nn.Module):
    """Thin wrapper around torchaudio's log-mel transform for DiT targets."""

    def __init__(self, sample_rate=16000, n_fft=1024, hop_length=256, n_mels=80):
        super().__init__()
        self.n_mels = n_mels
        self.hop_length = hop_length
        self.mel = torchaudio.transforms.MelSpectrogram(
            sample_rate=sample_rate,
            n_fft=n_fft,
            hop_length=hop_length,
            n_mels=n_mels,
            power=2.0,
        )

    def forward(self, wav):
        """wav: (batch, samples) -> log-mel (batch, frames, n_mels)."""
        if wav.dim() == 3:
            wav = wav.squeeze(1)
        mel = self.mel(wav)  # (batch, n_mels, frames)
        mel = torch.log(mel.clamp(min=1e-5))
        return mel.transpose(1, 2)  # (batch, frames, n_mels)
