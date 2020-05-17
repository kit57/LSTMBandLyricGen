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
        encoded = tokenizer.texts_to_sequences([generated_text])
        # pad or truncate text to the length that matches the trained model
        encoded = pad_sequences(encoded, maxlen=max_input_length)
        # predict from trained model, greedy search.
        y_ix = model.predict_classes(encoded, verbose=0)
        generated_text += ' '+ ix2word[y_ix[0]] # for example; y_ix = [66]

    print('<=== Seed_text: {} ===>'.format(seed_text))
    print('<=== Generated_lyrics: {} ===>\n'.format(generated_text))


if __name__ == "__main__":
    # Generator
    model = load_model('Wordmodel/lyrics_metal_model.h5')
    tokenizer = load(open('Wordmodel/saveWord/wordtokenizer.pkl', 'rb'))

    seed_text = 'we must be strong and we must be brave'
    generator(seed_text, tokenizer, 60, 4, model)

    seed_text = 'the wolf goddess'
    generator(seed_text, tokenizer, 50, 8, model)

    seed_text = 'fight for the king for the hammer and the ring'
    generator(seed_text, tokenizer, 40, 5, model)

    seed_text = 'freeze this night silhoutted'
    generator(seed_text, tokenizer, 50, 3, model)

    seed_text = 'fear'
    generator(seed_text, tokenizer, 50, 6, model)
    #############################################################################
    # # plot losses
    # loss_array = np.load('Wordmodel/saveWord/Wordloss.npz')['loss']
    # loss_train, loss_test = loss_array[0], loss_array[1]
    # loss_plot(loss_train, loss_test)