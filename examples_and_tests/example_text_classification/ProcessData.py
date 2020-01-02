# prepare your data
# use the IMDB dataset
import numpy as np
from keras.datasets import imdb  # differ


class ProcessData(object):

    def __init__(self):
        index_offset = 3
        (x_train, y_train), (x_test, y_test) = imdb.load_data(num_words=1000, index_from=index_offset)

        # I don't know why they reshape
        self.y_train = y_train.reshape(-1, 1)
        self.y_test = y_test.reshape(-1, 1)

        word_to_id = imdb.get_word_index()
        word_to_id = {k: (v + index_offset) for k, v in word_to_id.items()}
        word_to_id["<PAD>"] = 0
        word_to_id["<START>"] = 1
        word_to_id["<UNK>"] = 2
        id_to_word = {value: key for key, value in word_to_id.items()}

        # Converting the list of ids to sentences using the id_to_word correspondence
        x_test = list(map(lambda id_list: " ".join(id_to_word[id] for id in id_list), x_test))
        x_train = list(
            map(lambda id_list_singleton: " ".join(id_to_word[id] for id in id_list_singleton), x_train))

        self.x_train = np.array(x_train, dtype=np.str)
        self.x_test = np.array(x_test, dtype=np.str)

    def get_x_train(self):
        return self.x_train

    def get_y_train(self):
        return self.y_train



