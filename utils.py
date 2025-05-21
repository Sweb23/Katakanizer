import pandas as pd
from tensorflow.keras.preprocessing.text import Tokenizer


def create_tokenizers(is_light = False):
    df = pd.read_csv("loanwords.csv")

    # Shuffle and optionally reduce to 25%
    if is_light:
        df = df.sample(frac=0.25, random_state=42).reset_index(drop=True)

    # Extract input (Latin) and output (Katakana)
    romaji_texts = df["latin"].astype(str).tolist()
    katakana_texts = df["katakana"].astype(str).tolist()    

    # Tokenize Romaji
    romaji_tokenizer = Tokenizer(char_level=True)  # Tokenize at character level
    romaji_tokenizer.fit_on_texts(romaji_texts)

    # Tokenize Katakana
    katakana_tokenizer = Tokenizer(char_level=True)  # Tokenize at character level
    katakana_tokenizer.fit_on_texts(katakana_texts)

    romaji_sequences = romaji_tokenizer.texts_to_sequences(romaji_texts)
    romaji_vocab_size = len(romaji_tokenizer.word_index) + 1  # +1 for padding


    katakana_sequences = katakana_tokenizer.texts_to_sequences(katakana_texts)
    katakana_vocab_size = len(katakana_tokenizer.word_index) + 1

    # Padding sequences to max length
    max_seq_length = max(max(len(seq) for seq in romaji_sequences),
                        max(len(seq) for seq in katakana_sequences))
    
    return romaji_tokenizer, katakana_tokenizer, romaji_sequences, katakana_sequences, romaji_vocab_size, katakana_vocab_size, max_seq_length