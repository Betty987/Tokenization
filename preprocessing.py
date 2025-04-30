import nltk
import re

# Download required NLTK data
nltk.download('punkt')

# Step 1: Basic Text Cleaning
def clean_text(text):
    # Convert to lowercase
    text = text.lower()
    # Remove special characters and extra whitespace
    text = re.sub(r'\s+', ' ', text)  # Replace multiple spaces with single space
    text = re.sub(r'[^\w\s]', '', text)  # Remove punctuation
    return text.strip()

# Step 2: Tokenization with NLTK
tokenizer = nltk.word_tokenize


# Process Penn Treebank Dataset
try:
    # Load Penn Treebank dataset
    with open('/Users/bethtassew/Downloads/ptbdataset/ptb.train.txt', 'r', encoding='utf-8') as f:
        train_text = f.read()
    with open('/Users/bethtassew/Downloads/ptbdataset/ptb.valid.txt', 'r', encoding='utf-8') as f:
        val_text = f.read()
    with open('/Users/bethtassew/Downloads/ptbdataset/ptb.test.txt', 'r', encoding='utf-8') as f:
        test_text = f.read()

    # Clean the text
    train_text = clean_text(train_text)
    val_text = clean_text(val_text)
    test_text = clean_text(test_text)

    # Tokenize
    train_tokens = tokenizer(train_text)
    val_tokens = tokenizer(val_text)
    test_tokens = tokenizer(test_text)

    # Print results for Penn Treebank
    print("\n=== Penn Treebank Dataset ===")
    print(f"Training tokens (first 20): {train_tokens[:20]}")
    print(f"Validation tokens (first 20): {val_tokens[:20]}")
    print(f"Test tokens (first 20): {test_tokens[:20]}")
    print(f"Total training tokens: {len(train_tokens)}")
    print(f"Total validation tokens: {len(val_tokens)}")
    print(f"Total test tokens: {len(test_tokens)}")

except FileNotFoundError:
    print("Penn Treebank files not found. Skipping this dataset.")
