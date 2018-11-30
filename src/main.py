from __future__ import print_function
from keras.callbacks import LambdaCallback
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import LSTM 
from keras.optimizers import RMSprop
from keras.utils.data_utils import get_file
import numpy
import random
import sys
import io

file_name = input("Input a text file for the model to learn from. Format as 'filename.txt': ") 

with open(file_name) as f: 
    read_data = f.read().lower()
print('corpus length:', len(read_data))

chars = sorted(list(set(read_data)))
print('total chars:', len(chars))
char_index = dict((c, i) for i, c in enumerate(chars))
index_char = dict((i, c) for i, c in enumerate(chars))

maxlen = 40
step = 3
objects_array = []
next_char = []
for i in range(0, len(read_data) - maxlen, step):
    objects_array.append(read_data[i: i + maxlen])
    next_char.append(read_data[i + maxlen])
print('nb sequences:', len(objects_array))

print('Vectorization...')
x = numpy.zeros((len(objects_array), maxlen, len(chars)), dtype=numpy.bool)
y = numpy.zeros((len(objects_array), len(chars)), dtype=numpy.bool)
for i, objects in enumerate(objects_array):
    for t, char in enumerate(objects):
        x[i, t, char_index[char]] = 1
    y[i, char_index[next_char[i]]] = 1

print('Build Model...')
model = Sequential()
model.add(LSTM(128, input_shape=(maxlen, len(chars))))
model.add(Dense(len(chars), activation='softmax'))

optimizer = RMSprop(lr=0.01)
model.compile(loss='categorical_crossentropy', optimizer=optimizer)

def sample(predecessor, temperature=1.0):
    predecessor = numpy.asarray(predecessor).astype('float64')
    predecessor = numpy.log(predecessor) / temperature
    export_predecessor = numpy.exp(predecessor)
    predecessor = export_predecessor / numpy.sum(export_predecessor)
    probas = numpy.random.multinomial(1, predecessor, 1)
    return numpy.argmax(probas)

def every_epoch(epoch, _):
    print()
    print('---- Generating text after Epoch: %d' % epoch)

    start_index = random.randint(0, len(read_data) - maxlen - 1)
    for diversity in [0.2, 0.5, 1.0, 1.2]: 
        print('-----diversity:', diversity)

    generated = ''
    objects = read_data[start_index: start_index + maxlen]
    generated += objects
    print('-----Generating with seed: "' + objects + '"')
    sys.stdout.write(generated)

    for i in range(400):
        x_predecessor = numpy.zeros((1, maxlen, len(chars)))
        for t, char in enumerate(object_array):
            x_predecessor[0, t, char_index[char]] = 1.

        predecessor = model.predict(x_predecessor, verbose=0)[0]
        next_index = sample(predecessor, diversity)
        next_char = index_char[next_index]

        generated += next_char
        object_array = object_array[1:] + next_char

        sys.stdout.write(next_char)
        sys.stdout.flush()
    print()

print_callback = LambdaCallback(every_epoch=every_epoch)

model.fit(x,y,
    batch_size=128,
    epochs=60,
    callbacks=[print_callback])

