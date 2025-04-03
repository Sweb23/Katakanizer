from tensorflow import keras
import tensorflow as tf

def is_latin(text):
    return all('A' <= char <= 'Z' or 'a' <= char <= 'z' or char == '-' for char in text)


print("Loading model...\n")
model = keras.models.load_model("katakanizer_model.h5")

query = input("Enter your query in the latin alphabet : ")

if not is_latin(query):
    print("Error : input contains letters not recognized as latin.\n")
else:
    query = query.upper()

    result = model.predict(query)

    print("Your converted katakana sequence is : ", result)