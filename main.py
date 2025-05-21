from tensorflow import keras
from tensorflow.keras.preprocessing.sequence import pad_sequences
import utils
import numpy as np
import sys

def is_latin(text):
    return all('A' <= char <= 'Z' or 'a' <= char <= 'z' or char == '-' for char in text)

args = sys.argv

is_light = (args[1] == "light")

if is_light:
    model = keras.models.load_model("katakanizer_model_light.h5")
else:
    model = keras.models.load_model("katakanizer_model.h5")
    

print("Loading model...\n")

query = input("Enter your query in the latin alphabet : ")

if not is_latin(query):
    print("Error : input contains letters not recognized as latin.\n")
else:
    romaji_tokenizer, katakana_tokenizer, _, _, _, _, max_seq_length = utils.create_tokenizers(is_light)

    seq = romaji_tokenizer.texts_to_sequences([query.lower()])
    padded_seq = pad_sequences(seq, maxlen=max_seq_length, padding='post')

    # Predict
    prediction = model.predict([padded_seq, padded_seq])
    predicted_indices = np.argmax(prediction, axis=-1)[0]

    # Decode prediction
    output_katakana = ''.join([katakana_tokenizer.index_word.get(i, '') for i in predicted_indices if i != 0])

    print(f"Katakana: {output_katakana}")