"""
Qwen3-TTS: top-level assembly of the two model variants.

This file wires the shared pieces into two runnable models that mirror the two
paths in the technical report:

  Qwen3TTS12Hz  (RVQ + MTP path)
    text + speaker -> dual-track backbone -> codebook-0 head -> MTP fills
    residual RVQ codebooks -> MultiCodebookCodec.decode -> waveform.

  Qwen3TTS25Hz  (Flow-Matching path)
    text + speaker -> dual-track backbone -> single semantic-token head ->
    FlowMatchingDiT generates a mel from the token features -> MelVocoder ->
    waveform.

Both share `DualTrackBackbone` and `SpeakerEncoder`. The codecs here are used to
*produce token targets* from audio and to *decode tokens back to audio*; in this
learning setup they can be trained jointly or pre-trained, but the code keeps
the interfaces separate so each idea is visible on its own.

Running this file executes shape/smoke tests for both models (see __main__).
"""
import torch
import torch.nn as nn
import torch.nn.functional as F

from backbone import DualTrackBackbone, SpeakerEncoder
from codec import MultiCodebookCodec, SingleCodebookCodec, MelSpectrogram
from flow_dit import FlowMatchingDiT, MelVocoder
from layers import count_parameters
from mtp import MultiTokenPredictor


# Special speech-token ids used on the codebook-0 / semantic track. Real codes
# occupy [0, codebook_size); these markers live just above that range.
def _special_ids(codebook_size):
    return {"bos": codebook_size, "eos": codebook_size + 1}


class Qwen3TTS12Hz(nn.Module):
    """RVQ + Multi-Token-Prediction variant (the 12Hz family)."""

    def __init__(
        self,
        text_vocab_size,
        num_codebooks=8,
        codebook_size=1024,
        d_model=256,
        num_layers=6,
        num_heads=8,
        num_kv_heads=2,
        d_ff=768,
        max_seq_len=1024,
        speaker_dim=128,
        sample_rate=16000,
    ):
        super().__init__()
        self.num_codebooks = num_codebooks
        self.codebook_size = codebook_size
        self.special = _special_ids(codebook_size)
        # Codebook-0 track vocabulary = real codes + {BOS, EOS}.
        self.code0_vocab = codebook_size + 2

        self.speaker_encoder = SpeakerEncoder(embed_dim=speaker_dim)
        self.backbone = DualTrackBackbone(
            text_vocab_size=text_vocab_size,
            speech_vocab_size=self.code0_vocab,
            d_model=d_model,
            num_layers=num_layers,
            num_heads=num_heads,
            num_kv_heads=num_kv_heads,
            d_ff=d_ff,
            max_seq_len=max_seq_len,
            speaker_dim=speaker_dim,
        )
        # Head predicting the coarse codebook-0 (semantic) token.
        self.code0_head = nn.Linear(d_model, self.code0_vocab, bias=False)
        nn.init.normal_(self.code0_head.weight, std=0.02)
        # MTP fills in codebooks 1..NQ-1 for each frame.
        self.mtp = MultiTokenPredictor(d_model, num_codebooks, codebook_size)
        self.codec = MultiCodebookCodec(
            num_codebooks=num_codebooks, codebook_size=codebook_size, sample_rate=sample_rate
        )

    def _speech_input_embeds(self, code0_in):
        """Embed the codebook-0 input tokens for the backbone (teacher forcing)."""
        return self.backbone.speech_embedding(code0_in)

    def compute_loss(self, text_ids, codes, speaker_wav=None):
        """Training loss = codebook-0 LM loss + MTP loss over codebooks 1..NQ-1.

        text_ids: (b, text_len); codes: (b, frames, num_codebooks) from the
        codec; speaker_wav: (b, samples) reference audio or None.
        """
        b, frames, _ = codes.shape
        device = codes.device
        speaker_embed = self.speaker_encoder(speaker_wav) if speaker_wav is not None else None

        code0 = codes[..., 0]  # (b, frames)
        bos = torch.full((b, 1), self.special["bos"], dtype=torch.long, device=device)
        eos = torch.full((b, 1), self.special["eos"], dtype=torch.long, device=device)
        # Input track: BOS-in-backbone is the learned self.backbone.bos vector,
        # so the speech-token inputs are just code0 (teacher forced); targets
        # are code0 shifted left with EOS appended.
        code0_in = code0                                   # (b, frames)
        code0_tgt = torch.cat([code0, eos], dim=1)         # (b, frames+1)

        speech_embeds = self._speech_input_embeds(code0_in)
        hidden, prefix_len = self.backbone(text_ids, speech_embeds, speaker_embed)
        # Predictions for speech positions: from the BOS marker onward.
        speech_hidden = hidden[:, prefix_len - 1 :]        # (b, frames+1, d)
        code0_logits = self.code0_head(speech_hidden)       # (b, frames+1, vocab)
        loss_code0 = F.cross_entropy(
            code0_logits.reshape(-1, self.code0_vocab), code0_tgt.reshape(-1)
        )

        # MTP loss: use hidden states aligned to actual frames (drop the final
        # EOS-predicting position) to reconstruct every codebook of each frame.
        frame_hidden = speech_hidden[:, :frames].reshape(-1, hidden.shape[-1])  # (b*frames, d)
        flat_codes = codes.reshape(-1, self.num_codebooks)                       # (b*frames, NQ)
        mtp_logits = self.mtp(frame_hidden, flat_codes)
        loss_mtp = 0.0
        for k in range(self.num_codebooks):
            loss_mtp = loss_mtp + F.cross_entropy(mtp_logits[k], flat_codes[:, k])
        loss_mtp = loss_mtp / self.num_codebooks

        total = loss_code0 + loss_mtp
        return total, {"code0": loss_code0.item(), "mtp": loss_mtp.item()}

    @torch.no_grad()
    def generate(self, text_ids, speaker_wav=None, max_frames=200, temperature=1.0):
        """Autoregressively generate RVQ codes, then decode to a waveform.

        Returns (wav, codes). wav: (b, 1, samples); codes: (b, frames, NQ).
        """
        self.eval()
        b = text_ids.shape[0]
        device = text_ids.device
        speaker_embed = self.speaker_encoder(speaker_wav) if speaker_wav is not None else None

        # Build the fixed prefix once; regrow the speech track each step.
        code0_seq = torch.empty(b, 0, dtype=torch.long, device=device)
        all_codes = []
        finished = torch.zeros(b, dtype=torch.bool, device=device)

        for _ in range(max_frames):
            speech_embeds = (
                self._speech_input_embeds(code0_seq)
                if code0_seq.shape[1] > 0
                else torch.zeros(b, 0, self.backbone.d_model, device=device)
            )
            hidden, prefix_len = self.backbone(text_ids, speech_embeds, speaker_embed)
            last_hidden = hidden[:, -1]                     # (b, d) predicts next code0
            logits = self.code0_head(last_hidden) / temperature
            probs = F.softmax(logits, dim=-1)
            next_code0 = torch.multinomial(probs, 1).squeeze(-1)  # (b,)

            is_eos = next_code0 == self.special["eos"]
            finished = finished | is_eos
            if finished.all():
                break

            # Clamp specials to a valid code so MTP/codec never index OOB.
            safe_code0 = torch.clamp(next_code0, max=self.codebook_size - 1)
            frame_codes = self.mtp.generate_residuals(last_hidden, safe_code0, temperature)  # (b, NQ)
            all_codes.append(frame_codes)
            code0_seq = torch.cat([code0_seq, safe_code0.unsqueeze(1)], dim=1)

        if not all_codes:
            return torch.zeros(b, 1, 1, device=device), torch.zeros(b, 0, self.num_codebooks, device=device)
        codes = torch.stack(all_codes, dim=1)               # (b, frames, NQ)
        wav = self.codec.decode(codes)
        return wav, codes

    def num_parameters(self):
        return count_parameters(self)

    def save(self, filepath):
        torch.save({"config": self._config(), "state_dict": self.state_dict()}, filepath)
        print(f"Model saved to {filepath}")

    def _config(self):
        return {
            "text_vocab_size": self.backbone.text_vocab_size,
            "num_codebooks": self.num_codebooks,
            "codebook_size": self.codebook_size,
            "d_model": self.backbone.d_model,
        }

    @classmethod
    def load(cls, filepath, device="cpu"):
        ckpt = torch.load(filepath, map_location=device)
        model = cls(**ckpt["config"])
        model.load_state_dict(ckpt["state_dict"])
        model.to(device)
        print(f"Model loaded from {filepath}")
        return model


class Qwen3TTS25Hz(nn.Module):
    """Single-codebook + Flow-Matching DiT variant (the 25Hz family)."""

    def __init__(
        self,
        text_vocab_size,
        codebook_size=4096,
        d_model=256,
        num_layers=6,
        num_heads=8,
        num_kv_heads=2,
        d_ff=768,
        max_seq_len=1024,
        speaker_dim=128,
        n_mels=80,
        dit_dim=192,
        mel_hop=256,
        sample_rate=16000,
    ):
        super().__init__()
        self.codebook_size = codebook_size
        self.special = _special_ids(codebook_size)
        self.token_vocab = codebook_size + 2
        self.n_mels = n_mels
        self.mel_hop = mel_hop

        self.speaker_encoder = SpeakerEncoder(embed_dim=speaker_dim)
        self.backbone = DualTrackBackbone(
            text_vocab_size=text_vocab_size,
            speech_vocab_size=self.token_vocab,
            d_model=d_model,
            num_layers=num_layers,
            num_heads=num_heads,
            num_kv_heads=num_kv_heads,
            d_ff=d_ff,
            max_seq_len=max_seq_len,
            speaker_dim=speaker_dim,
        )
        self.token_head = nn.Linear(d_model, self.token_vocab, bias=False)
        nn.init.normal_(self.token_head.weight, std=0.02)

        self.codec = SingleCodebookCodec(codebook_size=codebook_size, sample_rate=sample_rate)
        # DiT conditions on backbone hidden states (d_model wide).
        self.dit = FlowMatchingDiT(n_mels=n_mels, d_model=dit_dim, d_ctx=d_model)
        self.vocoder = MelVocoder(n_mels=n_mels)
        self.mel_fn = MelSpectrogram(sample_rate=sample_rate, hop_length=mel_hop, n_mels=n_mels)

    def _lm_loss(self, text_ids, tokens, speaker_embed):
        """Next-semantic-token cross-entropy; returns (loss, token_hidden)."""
        b = tokens.shape[0]
        device = tokens.device
        eos = torch.full((b, 1), self.special["eos"], dtype=torch.long, device=device)
        tgt = torch.cat([tokens, eos], dim=1)
        speech_embeds = self.backbone.speech_embedding(tokens)
        hidden, prefix_len = self.backbone(text_ids, speech_embeds, speaker_embed)
        speech_hidden = hidden[:, prefix_len - 1 :]         # (b, frames+1, d)
        logits = self.token_head(speech_hidden)
        loss = F.cross_entropy(logits.reshape(-1, self.token_vocab), tgt.reshape(-1))
        return loss, speech_hidden[:, : tokens.shape[1]]    # hidden aligned to frames

    def compute_loss(self, text_ids, wav, speaker_wav=None):
        """Combined loss: LM next-token + flow-matching mel reconstruction.

        text_ids: (b, text_len); wav: (b, samples) the target audio;
        speaker_wav: reference audio (defaults to the target wav).
        """
        speaker_src = speaker_wav if speaker_wav is not None else wav
        speaker_embed = self.speaker_encoder(speaker_src)

        tokens = self.codec.encode(wav)                     # (b, frames)
        loss_lm, token_hidden = self._lm_loss(text_ids, tokens, speaker_embed)

        # Flow-matching decoder target: the log-mel of the true waveform. Token
        # features condition the DiT via cross-attention (no length matching
        # required), so we pass the per-token hidden states directly.
        mel = self.mel_fn(wav)                              # (b, T_mel, n_mels)
        loss_flow = self.dit.flow_loss(mel, token_hidden)

        total = loss_lm + loss_flow
        return total, {"lm": loss_lm.item(), "flow": loss_flow.item()}

    @torch.no_grad()
    def generate(self, text_ids, speaker_wav=None, max_frames=200, temperature=1.0, flow_steps=16):
        """Generate semantic tokens, run the DiT to a mel, vocode to a waveform."""
        self.eval()
        b = text_ids.shape[0]
        device = text_ids.device
        speaker_embed = self.speaker_encoder(speaker_wav) if speaker_wav is not None else None

        token_seq = torch.empty(b, 0, dtype=torch.long, device=device)
        hidden_frames = []
        finished = torch.zeros(b, dtype=torch.bool, device=device)

        for _ in range(max_frames):
            speech_embeds = (
                self.backbone.speech_embedding(token_seq)
                if token_seq.shape[1] > 0
                else torch.zeros(b, 0, self.backbone.d_model, device=device)
            )
            hidden, _ = self.backbone(text_ids, speech_embeds, speaker_embed)
            last_hidden = hidden[:, -1]
            logits = self.token_head(last_hidden) / temperature
            next_token = torch.multinomial(F.softmax(logits, dim=-1), 1).squeeze(-1)

            finished = finished | (next_token == self.special["eos"])
            if finished.all():
                break
            safe_token = torch.clamp(next_token, max=self.codebook_size - 1)
            hidden_frames.append(last_hidden)
            token_seq = torch.cat([token_seq, safe_token.unsqueeze(1)], dim=1)

        if not hidden_frames:
            return torch.zeros(b, 1, 1, device=device), torch.zeros(b, 0, device=device)

        token_hidden = torch.stack(hidden_frames, dim=1)    # (b, frames, d)
        frames = token_hidden.shape[1]
        # One mel frame per token frame times the codec/mel hop ratio. Here we
        # target `frames` mel steps (toy setting); scale up if desired.
        mel = self.dit.sample(token_hidden, mel_len=frames, steps=flow_steps)
        wav = self.vocoder(mel)
        return wav, token_seq

    def num_parameters(self):
        return count_parameters(self)

    def save(self, filepath):
        torch.save({"config": self._config(), "state_dict": self.state_dict()}, filepath)
        print(f"Model saved to {filepath}")

    def _config(self):
        return {
            "text_vocab_size": self.backbone.text_vocab_size,
            "codebook_size": self.codebook_size,
            "d_model": self.backbone.d_model,
            "n_mels": self.n_mels,
            "mel_hop": self.mel_hop,
        }

    @classmethod
    def load(cls, filepath, device="cpu"):
        ckpt = torch.load(filepath, map_location=device)
        model = cls(**ckpt["config"])
        model.load_state_dict(ckpt["state_dict"])
        model.to(device)
        print(f"Model loaded from {filepath}")
        return model


def _smoke_test():
    """Forward + generate on random data for both variants; prints param counts."""
    torch.manual_seed(0)
    text_vocab = 64
    b, text_len, sr = 2, 12, 16000
    wav_len = sr  # 1 second

    text_ids = torch.randint(0, text_vocab, (b, text_len))
    wav = torch.randn(b, wav_len) * 0.1

    print("=" * 60)
    print("Qwen3TTS12Hz (RVQ + MTP)")
    print("=" * 60)
    m12 = Qwen3TTS12Hz(text_vocab, num_codebooks=8, codebook_size=1024, d_model=256, num_layers=6)
    print(f"  params: {m12.num_parameters():,}")
    codes = m12.codec.encode(wav)  # (b, frames, NQ)
    print(f"  codec codes shape: {tuple(codes.shape)}")
    loss, parts = m12.compute_loss(text_ids, codes, speaker_wav=wav)
    print(f"  train loss: {loss.item():.4f}  parts: {parts}")
    gen_wav, gen_codes = m12.generate(text_ids, speaker_wav=wav, max_frames=20)
    print(f"  generated wav: {tuple(gen_wav.shape)}  codes: {tuple(gen_codes.shape)}")

    print("=" * 60)
    print("Qwen3TTS25Hz (Flow-Matching DiT)")
    print("=" * 60)
    m25 = Qwen3TTS25Hz(text_vocab, codebook_size=4096, d_model=256, num_layers=6)
    print(f"  params: {m25.num_parameters():,}")
    tokens = m25.codec.encode(wav)
    print(f"  codec tokens shape: {tuple(tokens.shape)}")
    loss, parts = m25.compute_loss(text_ids, wav, speaker_wav=wav)
    print(f"  train loss: {loss.item():.4f}  parts: {parts}")
    gen_wav, gen_tokens = m25.generate(text_ids, speaker_wav=wav, max_frames=20, flow_steps=4)
    print(f"  generated wav: {tuple(gen_wav.shape)}  tokens: {tuple(gen_tokens.shape)}")

    print("\nAll smoke tests passed.")


if __name__ == "__main__":
    _smoke_test()
