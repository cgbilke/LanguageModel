#this program runs the trained models from model.py to predict the next word 
from random import randint
from pickle import load
from keras.models import load_model
from keras.preprocessing.sequence import pad_sequences

#load the document we have already
def load_document(filename):
    file = open(filename, 'r')
    data = file.read()
    file.close()
    return data

#generate a sequence from our LSTM model
def generate_sequence(model, tokenizer, sequence_length, input_text, number_of_words):
    result = list()
    text = input_text
    for _ in range(number_of_words):
        encoded_sequence = tokenizer.texts_to_sequences([text])[0]
        encoded_sequence = pad_sequences([encoded_sequence], maxlen = sequence_length, truncating = 'pre')
        yhat = model.predict_classes(encoded_sequence, verbose = 0)
        output_word = ''
        for word, index in tokenizer.word_index.items():
            if index == yhat:
                output_word = word
                break
        input_text += ' ' + output_word
        result.append(output_word)
    return ' '.join(result)

#load the cleaned text file from model.py
input_filename = 'clean_words.txt'
document = load_document('clean_words.txt')
words = document.split('\n')
sequence_length = len(words[0].split()) - 1

#load the model and tokenizer from model.py
model = load_model('model.h1')
tokenizer = load(open('tokenizer.pk1', 'rb'))

#get a text from input to predict words from
input_text = input("Enter a word for the model to predict the next word(s) from: ")
print(input_text)

#generate the prediction from the generate_sequence helper function
prediction = generate_sequence(model, tokenizer, sequence_length, input_text, 1)
print(prediction)
