#this class cleans our document (preprocessing) and trains the model using the cleaned text document
from keras.preprocessing.text import Tokenizer
from keras.utils import to_categorical
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import LSTM
from keras.layers import Embedding
from keras.optimizers import RMSprop
from numpy import array
import string
from pickle import dump

input_filename = input("Please insert the name of a text file to train the model with. (format as filename.txt): ")

#load the document and get the data within
def load_document(filename):
    file = open(filename, 'r')
    data = file.read()
    file.close()
    return data

#helper function to clean the data into useable words without punctuation and non-alphabetics
def clean_data(doc):
    doc = doc.replace('--', ' ')
    words = doc.split()
    table = str.maketrans('', '', string.punctuation)
    words = [w.translate(table) for w in words]
    words = [word for word in words if word.isalpha()]
    words = [word.lower() for word in words]
    return words

#function to move our words into a new file w/ 1 word per line
def save_words(words, filename):
    data = '\n'.join(words)
    file = open(filename, 'w')
    file.write(data)
    file.close()

#load and clean our document for the model
document = load_document(input_filename)
words = clean_data(document)
print('Total Words: %d',  len(words))
print('Unique Words: %d',  len(set(words)))

length = 2
data_sequences = list()
for i in range(length, len(words)):
    sequence = words[i-length:i]
    line = ' '.join(sequence)
    data_sequences.append(line)
print('Total Sequences: %d',  len(data_sequences))

#save our new words to a file
output_filename = 'clean_words.txt'
save_words(data_sequences, output_filename)

#load our new document and point words to the new, cleaned document of data
document = load_document('clean_words.txt')
words = document.split('\n')

#encode the sequence of words as an integer using Tokenizer
tokenizer = Tokenizer()
tokenizer.fit_on_texts(words)
sequences = tokenizer.texts_to_sequences(words)
size_of_vocab = len(tokenizer.word_index) + 1

#separate our sequences into input and output
sequences = array(sequences)
X, y = sequences[:,:-1], sequences[:,-1]
y = to_categorical(y, num_classes=size_of_vocab)
length = X.shape[1]

#define our model to be trained
model = Sequential()
model.add(Embedding(size_of_vocab, 50, input_length=length))
model.add(LSTM(100, return_sequences = True))
model.add(LSTM(100))
model.add(Dense(100, activation = 'relu'))
model.add(Dense(size_of_vocab, activation = 'softmax'))
print(model.summary())

#compile and fit the model
optimizer = RMSprop(lr=0.01)
model.compile(loss = 'categorical_crossentropy', optimizer=optimizer, metrics = ['accuracy'])
model.fit(X, y, batch_size = 128, epochs = 100)

#save the model and tokenizer to be used after being trained
model.save('model.h1')
dump(tokenizer, open('tokenizer.pk1', 'wb'))
