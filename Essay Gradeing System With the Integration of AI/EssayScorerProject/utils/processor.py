def extract_features(text):
    """
    Extract features like word count, diversity, etc.
    """
    words = text.split()
    sentences = text.split('.')
    word_count = len(words)
    sentence_count = len([s for s in sentences if s.strip()])
    avg_word_length = sum(len(w) for w in words) / word_count if word_count > 0 else 0
    
    return {
        "word_count": word_count,
        "sentence_count": sentence_count,
        "avg_word_length": avg_word_length
    }
