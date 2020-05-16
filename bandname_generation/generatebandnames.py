
from pickle import load
from keras.models import load_model
from keras.utils import to_categorical
from keras.preprocessing.sequence import pad_sequences


def generator(seed_text, char2ix, ix2char, max_input_length, output_length, model):
    generated_text = seed_text
    # generate a fixed number of characters
    for _ in range(output_length):
        # integer-encoding seed_text
        encoded = [char2ix[char] for char in generated_text]
        # truncate sequences to a fixed length
        encoded = pad_sequences([encoded], maxlen=max_input_length, truncating='pre', value=0.0)[0]
        # one hot encoding
        encoded = to_categorical(encoded, num_classes=len(char2ix))
        # input of rnn requires to be 3d array
        encoded = encoded.reshape(encoded.shape[1], encoded.shape[0], 1)
        # predict from trained model, greedy search.
        # note that y_ix is nparray, e.g.[1]
        y_ix = model.predict_classes(encoded, verbose=0)
        # boolean of 1==np.array([1]) is true
        generated_text += ix2char[y_ix[0]]

    print('<=== Seed_text: {} ===>'.format(seed_text))
    print('<=== Generated_lyrics: {} ===>\n'.format(generated_text))




if __name__ == "__main__":
    # Generator
    model = load_model('model/bandGen.h5')
    char2ix, ix2char = load(open('model/save/dict.pkl', 'rb'))

    seed_text = "a"
    generator(seed_text, char2ix, ix2char, 32, 40, model)

    seed_text = 'th'
    generator(seed_text, char2ix, ix2char, 32, 40, model)

    seed_text = 'lose'
    generator(seed_text, char2ix, ix2char, 32, 40, model)

    seed_text = 'no'
    generator(seed_text, char2ix, ix2char, 32, 40, model)

