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
    print("Using light model.")
else:
    model = keras.models.load_model("katakanizer_model.h5")
    

print("Loading model...\n")
romaji_tokenizer, katakana_tokenizer, _, _, _, _, max_seq_length = utils.create_tokenizers(is_light)

while True:
    query = input("Enter your query in the latin alphabet : ")

    words = query.split()

    if any(not is_latin(word) for word in words):
        print("Error : input contains letters not recognized as latin.\n")
    else:
        result = []
        for word in words:
            seq = romaji_tokenizer.texts_to_sequences([word.lower()])
            padded_seq = pad_sequences(seq, maxlen=max_seq_length, padding='post')

            # Predict
            prediction = model.predict([padded_seq, padded_seq])
            predicted_indices = np.argmax(prediction, axis=-1)[0]

            output_katakana = ''.join([katakana_tokenizer.index_word.get(i, '') for i in predicted_indices if i != 0])
            result.append(output_katakana)

        print(f"Katakana: {' '.join(result)}")