from tokenizers import Tokenizer, models, trainers, pre_tokenizers
import os

# Create a directory to save the tokenizer
os.makedirs("tokenizer", exist_ok=True)

# Load the Penn Treebank dataset from local files
with open('/Users/bethtassew/Downloads/ptbdataset/ptb.train.txt', 'r', encoding='utf-8') as f:
    train_text = f.read()
with open('/Users/bethtassew/Downloads/ptbdataset/ptb.valid.txt', 'r', encoding='utf-8') as f:
    val_text = f.read()
with open('/Users/bethtassew/Downloads/ptbdataset/ptb.test.txt', 'r', encoding='utf-8') as f:
    test_text = f.read()

# Split training text into sentences (assuming sentences are separated by newlines)
train_texts = [line.strip() for line in train_text.split('\n') if line.strip()]

# Save training texts to a temporary file for tokenizer training
with open("ptb_train.txt", "w", encoding="utf-8") as f:
    for text in train_texts:
        f.write(text + "\n")

# Initialize a BPE tokenizer
tokenizer = Tokenizer(models.BPE())
tokenizer.pre_tokenizer = pre_tokenizers.Whitespace()  # Split on whitespace

# Configure the BPE trainer
trainer = trainers.BpeTrainer(
    vocab_size=30000,  # Adjust vocab size as needed
    min_frequency=2,   # Minimum frequency for merges
    special_tokens=["[UNK]", "[CLS]", "[SEP]", "[PAD]", "[MASK]"]
)

# Train the tokenizer on the text file
tokenizer.train(["ptb_train.txt"], trainer)

# Save the tokenizer
tokenizer.save("tokenizer/bpe_tokenizer.json")

# Tokenize a few sample sentences
sample_sentences = train_texts[:5]  # Take first 5 sentences
for sentence in sample_sentences:
    encoding = tokenizer.encode(sentence)
    print(f"Original: {sentence}")
    print(f"Tokens: {encoding.tokens}")
    print(f"Token IDs: {encoding.ids}")
    print()

# Clean up temporary file
os.remove("ptb_train.txt")
