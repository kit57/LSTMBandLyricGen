import os
import numpy as np
from pickle import load
from keras.models import load_model
from keras.preprocessing.sequence import pad_sequences

import tensorflow as tf


def generator(seed_text, tokenizer, max_input_length, output_length, model):
    ix2word = tokenizer.index_word
    generated_text = seed_text
    # generate a fixed number of words
    for _ in range(output_length):
        # encode the text as integer
        encoded = tokenizer.texts_to_sequences([seed_text])[0]  # [0] removes list of list problem
        # pad or truncate text to the length that matches the trained model
        encoded = pad_sequences([encoded], maxlen=max_input_length, truncating='pre', value=0.0)
        # predict from trained model, greedy search.
        y_ix = model.predict_classes(encoded, verbose=0)
        generated_text += ' ' + ix2word[y_ix[0]]  # e.g. y_ix = [66]

    print(' Seed_text: {} '.format(seed_text))
    print(' Generated_lyrics: {} ===>\n'.format(generated_text))


if __name__ == "__main__":
    # Generator
    model = load_model('Wordmodel/lyrics_metal_model.h5')
    tokenizer = load(open('Wordmodel/saveWord/wordtokenizer.pkl', 'rb'))

    seed_text = 'who is really slim shady'
    generator(seed_text, tokenizer, 20, 10, model)

    seed_text = 'never'
    generator(seed_text, tokenizer, 20, 10, model)

    #############################################################################
    # # plot losses
    # loss_array = np.load('Wordmodel/saveWord/Wordloss.npz')['loss']
    # loss_train, loss_test = loss_array[0], loss_array[1]
    # loss_plot(loss_train, loss_test)