
import os
import numpy as np
import gensim.models.keyedvectors as word2vec
from pickle import dump
import keras
from keras.preprocessing.text import Tokenizer
from keras.utils import to_categorical
from keras.models import Sequential, load_model
from keras.layers import Dense, LSTM, Embedding, Dropout, Activation
from keras.callbacks import ModelCheckpoint, Callback
import tensorflow as tf
import warnings
warnings.filterwarnings("ignore")


class wordlevel_model():

    def __init__(self, file_path, length, epoch, batch_size):
        self.file_path = file_path
        self.epoch = epoch
        self.batch_size = batch_size
        self.length = length

    def get_data(self):

        max_length = 10
        try:
            self.corpus = open(self.file_path, 'r', encoding='utf8').read().lower().split('\n\n')
            # Cut encoded doc into 'self.length+1' long pairs
            #pairs = [''.join(self.corpus[i-0:i+1]) for i in range(0,max_length)]
        except UnicodeDecodeError:
            import codecs

            self.corpus = codecs.open(self.file_path, 'r', encoding='utf8').read().lower().split('\n\n')
            # Cut encoded doc into 'self.length+1' long pairs
            #pairs = [''.join(self.corpus[i-0:i+1]) for i in range(0,max_length)]

        #self.corpus= self.clean_text()

        self.tokenizer= Tokenizer() # Initialize Tokenizer
        self.tokenizer.fit_on_texts(self.corpus) # get index for words

        # Example:
        # sequences: [[1, 2, 3, 4, 6, 7], [1, 3]]
        # word_index: {'the': 1, 'earth': 2, 'is': 3, 'an': 4, 'awesome': 5, 'place': 6, 'live': 7}
        encoded_pairs = self.tokenizer.texts_to_sequences(self.corpus)  # Integer encode
        # vocabulary size
        self.VOCAB_SIZE = len(self.tokenizer.word_index) + 1  # +1 as index starts from 1(not 0)
        self.word_index = self.tokenizer.word_index
        # ix2word -> tokenizer.index_word, word2ix -> tokenizer.word_index
        dump(self.tokenizer, open('Wordmodel/saveWord/wordtokenizer.pkl', 'wb'))

        return encoded_pairs

    def prepare_data(self):

        encoded_pairs = self.get_data()

        # separate each line into input and output
        pairs = np.array(encoded_pairs)
        pairs = keras.preprocessing.sequence.pad_sequences(pairs, maxlen=30)
        # X has length long characters, y is the last character
        self.X = pairs[:, 0: -1] # X: 10words
        self.y = pairs[:, -1] # y: 1 word
        # one hot encoding targets to fit 'categorical_crossentropy'
        self.y = to_categorical(self.y, num_classes=self.VOCAB_SIZE)
        #print('data prepared')
        np.savez_compressed('Wordmodel/saveWord/Worddata.npz', X_array=self.X, y_array=self.y)


    def train_model(self):

        self.prepare_data()
        self.LSTM_model()

    def LSTM_model(self):
        '''Define the LSTM network'''
        embedding_matrix = self.embeddings()

        self.model = Sequential() # Embedding layer
        self.model.add(Embedding(input_dim=len(self.word_index)+1, output_dim=embedding_matrix.shape[1],
                            weights=[embedding_matrix], trainable=False))
        self.model.add(LSTM(units=embedding_matrix.shape[1], kernel_initializer='he_normal'))
        self.model.add(Dropout(0.2))
        self.model.add(Dense(units=embedding_matrix.shape[0],activation='relu', kernel_initializer='he_normal'))
        self.model.add(Activation('softmax'))

        #
        # # Remove weights to disable using pre-trained word embedding
        # self.model.add(Embedding(embedding_matrix.shape[0], embedding_matrix.shape[1],
        #                          weights=[embedding_matrix], input_length=self.length, trainable=False))
        # # If True: return c_t and h_t to next layer
        # self.model.add(LSTM(220, return_sequences=True, input_shape=(self.length+1, 1), kernel_initializer='he_normal'))
        # self.model.add(Dropout(0.2))
        # self.model.add(LSTM(220, return_sequences=False, kernel_initializer='he_normal'))
        # self.model.add(Dropout(0.2))
        # self.model.add(Dense(330, activation='relu', kernel_initializer='he_normal'))
        # self.model.add(Dropout(0.25))
        # self.model.add(Dense(embedding_matrix.shape[0], activation='softmax'))

        # setting up parameters
        self.model.compile(loss='categorical_crossentropy', optimizer='RMSprop', metrics=['accuracy'])
        print(self.model.summary())

        early_stopping = keras.callbacks.EarlyStopping(monitor='loss', mode='min', verbose=1, patience=5)
        history = LossHistory()

        weights = ModelCheckpoint(filepath='Wordmodel/lyrics_metal_model.h5')

        self.model.fit(self.X,
                       self.y,
                       batch_size=self.batch_size,
                       epochs=self.epoch,
                       validation_split=0.25,
                       callbacks=[history, weights, early_stopping]
                       )


    def embeddings(self):
        '''
        Match pretrained embedding and dataset embedding -> unique_words x emb_size
        '''

        glove_ptr_path = '../data/glove.6B.200d.txt'  # Load pretrained model.
        EMBED_DIM = 200  # Dimension
        # Load pretrained model (as intermediate data is not included, the model cannot be refined with additional data)

        embeddings_index={}
        f = open(glove_ptr_path, encoding='utf8')
        for line in f:
            values = line.split()
            word = values[0]
            coefs = np.asarray(values[1:], dtype='float32')
            embeddings_index[word] = coefs
        f.close()

        embedding_matrix = np.zeros((len(self.word_index) + 1, EMBED_DIM))
        for word, i in self.word_index.items():
            embedding_vector = embeddings_index.get(word)
            if embedding_vector is not None:
                # words not found in embedding index will be all-zeros.
                embedding_matrix[i] = embedding_vector

        # glove_ptr = word2vec.KeyedVectors.load_word2vec_format(glove_ptr_path, binary=True)
        # # glove_dict={}
        # # for line in glove_ptr.index2word:
        # #     values = line
        # #     word = values[0]
        # #     coefs = np.asarray(values[1:], dtype='float32')
        # #     glove_dict[word] = coefs
        #
        # print('Initialize embedding matrix')
        # # initialize embedding matrix for our dataset
        # embedding_matrix = np.zeros((self.VOCAB_SIZE, EMBED_DIM))
        # # count words that appear only in the dataset. word_index.items() yields dict of word:index pair
        # for word, ix in self.tokenizer.word_index.items():
        #     embedding_vector = glove_ptr.wv[glove_ptr.wv.index2word[ix]]
        #     if embedding_vector is not None:
        #         # words not found in glove matrix will be all-zeros.
        #         embedding_matrix[ix] = embedding_vector

        return embedding_matrix


    def resume_training(self):

        self.get_pretrained_data()
        self.LSTM_model()

    def get_pretrained_data(self):

        # load pretrained model
        self.model = load_model('Wordmodel/lyrics_metal_model.h5')
        # load cleaned data
        data = np.load('Wordmodel/saveWord/Worddata.npz')
        self.X = data['X_array']
        self.y = data['y_array']
        self.VOCAB_SIZE = self.VOCAB_SIZE

# customize an History class that save losses to a file for each epoch
class LossHistory(Callback):

    def on_train_begin(self, logs=None):

        if os.path.exists('Wordmodel/saveWord/Wordloss.npz'):
            self.loss_array = np.load('Wordmodel/saveWord/Wordloss.npz')['loss']
        else:
            self.loss_array = np.empty([2, 0])

    def on_epoch_end(self, epoch, logs=None):
        # append new losses to loss_array
        loss_train = logs.get('loss')
        loss_test = logs.get('val_loss')

        loss_new = np.array([[loss_train], [loss_test]])  # 2 x 1 array
        self.loss_array = np.concatenate((self.loss_array, loss_new), axis=1)
        # save model to disk
        np.savez_compressed('Wordmodel/saveWord/Wordloss.npz', loss=self.loss_array)

if __name__ == "__main__":

    file_path = '../data/lyrics_dataset/heavymetallyrics.txt'

    length = 25 # fixed length of input sequences
    epoch = 315
    batch_size = 1024

    model = wordlevel_model(file_path, length, epoch, batch_size)
    model.train_model()
    # train or resume train
    # if not os.path.exists('Wordmodel/lyrics_metal_model.h5'):
    #     print('<==========| Data preprocessing... |==========>')
    #     model.train_model()

    # else:
    #     print('<==========| Resume training... |==========>')
    #     model.resume_training()