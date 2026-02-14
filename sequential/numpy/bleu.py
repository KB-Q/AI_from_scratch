"""
Calculate the BLEU score for a given translation and reference text.
"""

def bleu_score(translation, reference, max_n=4):
    """
    Calculate BLEU score for a translation against a reference.
    
    Args:
        translation: The translated text (string)
        reference: The reference text (string)
        max_n: Maximum n-gram size to consider (default: 4)
    
    Returns:
        float: BLEU score between 0 and 1
    """
    import math
    from collections import Counter
    
    # Step 1: Tokenize translation and reference
    translation_tokens = translation.lower().split()
    reference_tokens = reference.lower().split()
    
    # Step 2: Calculate brevity penalty
    if len(translation_tokens) == 0: return 0.0
    if len(translation_tokens) > len(reference_tokens):
        brevity_penalty = 1.0
    else:
        brevity_penalty = math.exp(1 - len(reference_tokens) / len(translation_tokens))
    
    # Step 3: Calculate precision for each n-gram size
    precisions = []
    
    for n in range(1, max_n + 1):
        ngram_length = len(translation_tokens) - n + 1
        
        # Generate n-grams for translation
        translation_ngrams = []
        for i in range(ngram_length):
            ngram = tuple(translation_tokens[i:i + n])
            translation_ngrams.append(ngram)
        
        # Generate n-grams for reference
        reference_ngrams = []
        for i in range(ngram_length):
            ngram = tuple(reference_tokens[i:i + n])
            reference_ngrams.append(ngram)
        
        # If no n-grams exist, skip this n
        if len(translation_ngrams) == 0:
            precisions.append(0.0)
            continue
        
        # Count n-grams with clipping (modified precision)
        translation_ngram_counts = Counter(translation_ngrams)
        reference_ngram_counts = Counter(reference_ngrams)
        
        clipped_counts = 0
        for ngram, count in translation_ngram_counts.items():
            clipped_counts += min(count, reference_ngram_counts.get(ngram, 0))
        
        # Calculate precision for this n-gram size
        precision = clipped_counts / len(translation_ngrams)
        precisions.append(precision)
    
    # Step 4: Calculate geometric mean of precisions
    # Filter out zero precisions to avoid log(0)
    non_zero_precisions = [p for p in precisions if p > 0]
    
    if len(non_zero_precisions) == 0:
        return 0.0
    
    # Geometric mean using log space for numerical stability
    log_precision_sum = sum(math.log(p) for p in non_zero_precisions)
    geometric_mean = math.exp(log_precision_sum / len(precisions))
    
    # Step 5: Calculate final BLEU score
    bleu = brevity_penalty * geometric_mean
    return bleu

if __name__ == "__main__":
    translation = "The cat sat on the mat"
    reference = "The cat sat on the mat"
    print(bleu_score(translation, reference))

    translation = "The cat sat on the mat"
    reference = "The mat sat on the cat"
    print(bleu_score(translation, reference))