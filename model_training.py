from sklearn.model_selection import train_test_split
import numpy as np
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, LSTM, Dense, Embedding
import utils
import sys

args = sys.argv

is_light = (args[1] == "light")

if is_light:
    print("Training light model\n")

romaji_tokenizer, katakana_tokenizer, romaji_sequences, katakana_sequences, romaji_vocab_size, katakana_vocab_size, max_seq_length = utils.create_tokenizers(is_light)

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
# Disabling automatic masking to avoid NotEqual
encoder_inputs = Input(shape=(max_seq_length,))
encoder_embedding = Embedding(input_dim=romaji_vocab_size, output_dim=128, mask_zero=False)(encoder_inputs)
encoder_lstm = LSTM(256, return_state=True)
encoder_outputs, state_h, state_c = encoder_lstm(encoder_embedding)

# Decoder
# Disabling automatic masking to avoid NotEqual
decoder_inputs = Input(shape=(max_seq_length,))
decoder_embedding = Embedding(input_dim=katakana_vocab_size, output_dim=128, mask_zero=False)(decoder_inputs)
decoder_lstm = LSTM(256, return_sequences=True, return_state=True)
decoder_outputs, _, _ = decoder_lstm(decoder_embedding, initial_state=[state_h, state_c])
decoder_dense = Dense(katakana_vocab_size, activation="softmax")
decoder_outputs = decoder_dense(decoder_outputs)

# Define the full model
model = Model([encoder_inputs, decoder_inputs], decoder_outputs)
model.compile(optimizer="adam", loss="categorical_crossentropy", metrics=["accuracy"])

# Train the model
model.fit([X_train, X_train], y_train, batch_size=64, epochs=30, validation_data=([X_test, X_test], y_test))

if is_light:
    model.save("katakanizer_model_light.h5")
else:
    model.save("katakanizer_model.h5")