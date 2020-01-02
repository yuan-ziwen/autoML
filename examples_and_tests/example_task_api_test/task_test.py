import autokeras as ak
import tensorflow as tf
import pytest
import numpy as np


def imdb_raw(num_instances=1800):
    index_offset = 3
    (x_train, y_train), (x_test, y_test) = tf.keras.datasets.imdb.load_data(num_words=1000, index_from=index_offset)

    x_train = x_train[:num_instances]
    # I don't know why reshape them
    y_train = y_train[:num_instances].reshape(-1, 1)
    y_test = y_test[:num_instances].reshape(-1, 1)
    x_test = x_train[:num_instances]

    print("can you see me")
    word_to_id = tf.keras.datasets.imdb.get_word_index()
    word_to_id = {k: (v + index_offset) for k, v in word_to_id.items()}
    word_to_id["<PAD>"] = 0
    word_to_id["<START>"] = 1
    word_to_id["<UNK>"] = 2
    id_to_word = {value: key for key, value in word_to_id.items()}

    print("i am here")
    # Converting the list of ids to| sentences using the id_to_word correspondence
    x_test = list(map(lambda id_list: " ".join(id_to_word[id] for id in id_list), x_test))
    x_train = list(map(lambda id_list_singleton: " ".join(id_to_word[id] for id in id_list_singleton), x_train))

    x_train = np.array(x_train, dtype=np.str)
    x_test = np.array(x_test, dtype=np.str)

    return (x_train, y_train), (x_test, y_test)


@pytest.fixture(scope="module")
def tmp_dir(tmpdir_factory):
    return tmpdir_factory.mktemp("task_api")


def test_text_classifier():
    (train_x, train_y), (test_x, test_y) = imdb_raw()
    clf = ak.TextClassifier(verbose=True)  # differ
    clf.fit(train_x, train_y)  # differ
    print(clf.predict(test_x).shape == (len(test_x), 1))
    print(clf.evaluate(test_x, test_y))


test_text_classifier()
