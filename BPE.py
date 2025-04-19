# Install tokenizers library
!pip install tokenizers

import matplotlib.pyplot as plt
import numpy as np
from tokenizers import Tokenizer, models, trainers, pre_tokenizers
from tokenizers.pre_tokenizers import Whitespace
from IPython.display import Image

def train_bpe_tokenizer(corpus, vocab_size=100):
    """Train a BPE tokenizer on the given corpus."""
    # Initialize a BPE tokenizer
    tokenizer = Tokenizer(models.BPE(unk_token="[UNK]"))
    
    # Use whitespace pre-tokenization to split on spaces
    tokenizer.pre_tokenizer = Whitespace()
    
    # Train the tokenizer
    trainer = trainers.BpeTrainer(vocab_size=vocab_size, special_tokens=["[UNK]"])
    tokenizer.train_from_iterator(corpus, trainer)
    
    return tokenizer

def create_char_tokenizer():
    """Create a character-level tokenizer."""
    # Initialize a WordPiece model (used as a base, but we'll override with char-level logic)
    tokenizer = Tokenizer(models.WordPiece(unk_token="[UNK]"))
    
    # Custom pre-tokenizer to split into characters
    tokenizer.pre_tokenizer = pre_tokenizers.CharDelimiterSplit(delimiter='')
    
    # Since we're doing char-level, no training is needed; use a small vocab
    return tokenizer

def tokenize_and_visualize(sentence, tokenizer, title):
    """Tokenize the sentence and visualize the tokens."""
    # Encode the sentence
    encoding = tokenizer.encode(sentence)
    tokens = encoding.tokens
    
    # Create a color map for tokens
    colors = plt.cm.tab20(np.linspace(0, 1, len(set(tokens))))
    token_to_color = {token: colors[i % len(colors)] for i, token in enumerate(set(tokens))}
    
    # Plot setup
    plt.figure(figsize=(12, 2))
    x, y = 0, 0
    for token in tokens:
        plt.text(x, y, token, bbox=dict(facecolor=token_to_color[token], alpha=0.5))
        x += len(token) * 0.15  # Adjust spacing based on token length
    
    plt.title(title)
    plt.axis('off')
    plt.savefig(f'{title.lower().replace(" ", "_")}.png')
    plt.close()
    
    return tokens

def main():
    # Example sentence
    sentence = "The quick brown fox jumps"
    
    # Create a small corpus (including the sentence) for BPE training
    corpus = [
        sentence,
        "The fox runs fast",
        "A quick dog jumps high",
        "Brown foxes climb hills"
    ]
    
    # Train BPE tokenizer
    bpe_tokenizer = train_bpe_tokenizer(corpus, vocab_size=100)
    
    # Create character-level tokenizer
    char_tokenizer = create_char_tokenizer()
    
    # Tokenize and visualize for BPE
    bpe_tokens = tokenize_and_visualize(sentence, bpe_tokenizer, "BPE Tokenization")
    
    # Tokenize and visualize for character-level
    char_tokens = tokenize_and_visualize(sentence, char_tokenizer, "Character-Level Tokenization")
    
    # Print results
    print(f"Original Sentence: {sentence}")
    print(f"\nBPE Tokens: {bpe_tokens}")
    print(f"Character-Level Tokens: {char_tokens}")
    
    # Display the saved images
    print("\nVisualizations:")
    display(Image('bpe_tokenization.png'))
    display(Image('character-level_tokenization.png'))

if __name__ == "__main__":
    main()
