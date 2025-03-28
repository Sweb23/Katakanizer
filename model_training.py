from sklearn.model_selection import train_test_split
import pandas as pd
import numpy as np
import tensorflow as tf
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, LSTM, Dense, Embedding

# Load dataset
df = pd.read_csv("loanwords.csv")

# Extract input (Latin) and output (Katakana)
romaji_texts = df["latin"].astype(str).tolist()
katakana_texts = df["katakana"].astype(str).tolist()

# Tokenize Romaji
romaji_tokenizer = Tokenizer(char_level=True)  # Tokenize at character level
romaji_tokenizer.fit_on_texts(romaji_texts)
romaji_sequences = romaji_tokenizer.texts_to_sequences(romaji_texts)
romaji_vocab_size = len(romaji_tokenizer.word_index) + 1  # +1 for padding

# Tokenize Katakana
katakana_tokenizer = Tokenizer(char_level=True)  # Tokenize at character level
katakana_tokenizer.fit_on_texts(katakana_texts)
katakana_sequences = katakana_tokenizer.texts_to_sequences(katakana_texts)
katakana_vocab_size = len(katakana_tokenizer.word_index) + 1

# Padding sequences to max length
max_seq_length = max(max(len(seq) for seq in romaji_sequences),
                     max(len(seq) for seq in katakana_sequences))

romaji_padded = pad_sequences(romaji_sequences, maxlen=max_seq_length, padding="post")
katakana_padded = pad_sequences(katakana_sequences, maxlen=max_seq_length, padding="post")

# One-hot encode the Katakana outputs
katakana_padded_onehot = to_categorical(katakana_padded, num_classes=katakana_vocab_size)

print("INFO : Sequences are converted to 16-bit floats.\n")
romaji_padded = romaji_padded.astype(np.float16)
katakana_padded_onehot = katakana_padded_onehot.astype(np.float16)

print("Splitting training and testing data...\n")
X_train, X_test, y_train, y_test = train_test_split(romaji_padded, katakana_padded_onehot, test_size=0.2, random_state=42)

# Encoder
encoder_inputs = Input(shape=(max_seq_length,))
encoder_embedding = Embedding(input_dim=romaji_vocab_size, output_dim=128, mask_zero=True)(encoder_inputs)
encoder_lstm = LSTM(256, return_state=True)
encoder_outputs, state_h, state_c = encoder_lstm(encoder_embedding)

# Decoder
decoder_inputs = Input(shape=(max_seq_length,))
decoder_embedding = Embedding(input_dim=katakana_vocab_size, output_dim=128, mask_zero=True)(decoder_inputs)
decoder_lstm = LSTM(256, return_sequences=True, return_state=True)
decoder_outputs, _, _ = decoder_lstm(decoder_embedding, initial_state=[state_h, state_c])
decoder_dense = Dense(katakana_vocab_size, activation="softmax")
decoder_outputs = decoder_dense(decoder_outputs)

# Define the full model
model = Model([encoder_inputs, decoder_inputs], decoder_outputs)
model.compile(optimizer="adam", loss="categorical_crossentropy", metrics=["accuracy"])

# Train the model
model.fit([X_train, X_train], y_train, batch_size=64, epochs=40, validation_data=([X_test, X_test], y_test))

model.save("katakanizer_model.h5")