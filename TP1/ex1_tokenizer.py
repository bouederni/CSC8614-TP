from transformers import GPT2Tokenizer

tokenizer = GPT2Tokenizer.from_pretrained("gpt2")
phrase = "Artificial intelligence is metamorphosing the world!"
phrase2 = "GPT models use BPE tokenization to process unusual words like antidisestablishmentarianism."

def func_2a():
    phrase = "Artificial intelligence is metamorphosing the world!"
    tokens = tokenizer.tokenize(phrase)

    print(tokens)

def func_2b():
    token_ids = tokenizer.encode(phrase)
    print("Token IDs:", token_ids)

    print("Détails par token:")
    for tid in token_ids:
        txt = tokenizer.decode([tid])
        print(tid, repr(txt))

def func_2c():
    tokens2 = tokenizer.tokenize(phrase2)
    words = phrase2.split()
    long_word = max(words, key=len)
    long_word_tokens = tokenizer.tokenize(long_word)

    print(tokens2)
    print(f"\nNombre de sous-tokens pour '{phrase2}' :", len(tokens2))

    print("\nTokens du mot le plus long : " + str(long_word_tokens))
    print("\nNombre de tokens du mot le plus long : " + str(len(long_word_tokens)))





func_2c()