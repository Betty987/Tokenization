from tokenizers import Tokenizer, models, trainers, pre_tokenizers

# Input sentence
sentence = "The quick brown fox jumps"

# BPE Tokenization
bpe_tokenizer = Tokenizer(models.BPE(unk_token="[UNK]"))
bpe_tokenizer.pre_tokenizer = pre_tokenizers.Whitespace()
trainer = trainers.BpeTrainer(vocab_size=100, special_tokens=["[UNK]"])
bpe_tokenizer.train_from_iterator([sentence, "The fox runs fast"], trainer)
bpe_tokens = bpe_tokenizer.encode(sentence).tokens



# Print results
print(f"Original Sentence: {sentence}")
print(f"BPE Tokens: {bpe_tokens}")
