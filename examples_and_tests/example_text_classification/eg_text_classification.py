# prepare your data
# use the IMDB dataset
import numpy as np
from keras.datasets import imdb  # differ
import autokeras as ak

index_offset = 3
(x_train, y_train), (x_test, y_test) = imdb.load_data(num_words=1000, index_from=index_offset)

# I don't know why reshape them
y_train = y_train.reshape(-1, 1)
y_test = y_test.reshape(-1, 1)

print("can you see me")
word_to_id = imdb.get_word_index()
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

print("i am going to run the textclassifier")
####run the TextClassifier
clf = ak.TextClassifier(verbose=True)  # try 10 models maximum # differ
# verbose is a boolean, setting it o true prints to stdout.
#differ: remove the "max_trails" attribute, cuz my autokeras version TextClassifier class does not have this attribute.
print("i am here can you see me?")


clf.fit(x_train, y_train)
print("hi?")
predicted_y = clf.predict(x_test)
print("hello from outside")
print(clf.evaluate(x_test, y_test))
