
import keras
from keras.utils import np_utils
from keras.models import Sequential, load_model
from keras.layers.core import Dense, Activation, Dropout
from keras.layers.recurrent import LSTM
from keras.callbacks import ModelCheckpoint, Callback

import numpy as np
from pickle import dump
import string, os


class char_model():

    def __init__(self, file_path, length, epoch, batch_size):
        self.file_path = file_path
        self.epoch = epoch
        self.batch_size = batch_size
        self.length = length

    def get_data(self):

        try:
            corpus = open(self.file_path).read().lower()
        except UnicodeDecodeError:
            import codecs
            corpus = codecs.open(self.file_path, encoding='utf8').read().lower()

        corpus = self.clean_text(corpus=corpus)

        chars = list(set(corpus))
        self.VOCAB_SIZE = len(chars)
        char_to_ix = {char: ix for ix, char in enumerate(chars)} # char:index
        ix_to_char = {ix: char for ix, char in enumerate(chars)} # index:char

        combined = [char_to_ix, ix_to_char]
        dump(combined, open('model/save/dict.pkl', 'wb'))

        return corpus, char_to_ix, ix_to_char, chars

    def prepare_data(self):
        corpus, char_to_ix, ix_to_char, chars = self.get_data()
        # prepare the dataset of input to output pairs encoded as integers
        dataX = []
        dataY = []
        seq_lenght=self.length
        for i in range(0, (len(corpus) - seq_lenght), 1):
            seq_input = corpus[i:i + seq_lenght]
            seq_output = corpus[i + seq_lenght]
            dataX.append([char_to_ix[char] for char in seq_input])
            dataY.append(char_to_ix[seq_output])
        n_patterns = len(dataX)

        # reshape X to be [samples, time steps, features]
        self.X = np.reshape(dataX, (n_patterns, seq_lenght, 1))
        # normalize
        self.X = self.X / float(self.VOCAB_SIZE)
        # one hot encode the output variable
        self.y = np_utils.to_categorical(dataY)

        # save data in a npz file
        np.savez_compressed('model/save/data.npz', X_array=self.X, y_array=self.y)


    def clean_text(self, corpus):

        corpus = "".join(v for v in corpus if v not in string.punctuation).lower()
        corpus = corpus.encode("utf8").decode("ascii", 'ignore')
        return corpus

    def train_model(self):

        self.prepare_data()
        self.LSTM_model()

    def LSTM_model(self):

        # Create the Keras LSTM structure
        N_UNITS = 150

        self.model = Sequential()
        self.model.add(LSTM(N_UNITS, return_sequences=True, input_shape=(self.length, self.VOCAB_SIZE), kernel_initializer='he_normal'))
        self.model.add(Dropout(0.2))
        self.model.add(LSTM(N_UNITS, return_sequences=False, kernel_initializer='he_normal'))
        self.model.add(Dropout(0.2))
        self.model.add(Dense(self.VOCAB_SIZE))
        # model.add(Dense(1000))
        self.model.add(Activation('softmax'))

        customAdam = keras.optimizers.Adam(lr=0.0001)
        self.model.compile(optimizer=customAdam,
                      loss="binary_crossentropy",
                      metrics=["mean_squared_error","binary_crossentropy"])

        early_stopping = keras.callbacks.EarlyStopping(monitor='loss', mode='min', verbose=1, patience=5)
        print(self.model.summary())

        history = LossHistory()

        weights = ModelCheckpoint(filepath='model/bandGen.h5')

        self.model.fit(self.X,
                  self.y,
                  batch_size=self.batch_size,
                  epochs=self.epoch,
                  validation_split=0.25,
                  callbacks=[history, weights, early_stopping]
                  )

        print(history)




# customize an History class that save losses to a file for each epoch
class LossHistory(Callback):

    def on_train_begin(self, logs=None):

        if os.path.exists('model/save/loss.npz'):
            self.loss_array = np.load('model/save/loss.npz')['loss']
        else:
            self.loss_array = np.empty([2, 0])

    def on_epoch_end(self, epoch, logs=None):
        # append new losses to loss_array
        loss_train = logs.get('loss')
        loss_test = logs.get('val_loss')

        loss_new = np.array([[loss_train], [loss_test]])  # 2 x 1 array
        self.loss_array = np.concatenate((self.loss_array, loss_new), axis=1)
        # save to disk
        np.savez_compressed('model/save/loss.npz', loss=self.loss_array)

if __name__ == "__main__":

    file_path = '../data/metal_dataset/metal_bands.txt'

    length = 32
    epoch = 20
    batch_size = 1024

    model = char_model(file_path, length, epoch, batch_size)

    print('<==========| Data preprocessing... |==========>')
    model.train_model()

