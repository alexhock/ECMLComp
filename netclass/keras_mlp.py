import csv
import random
import numpy as np
import matplotlib.pyplot as plt
from sklearn.cross_validation import StratifiedKFold
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import LabelEncoder
from sklearn import metrics
from sklearn.ensemble import RandomForestClassifier, ExtraTreesClassifier, GradientBoostingClassifier
from sklearn.naive_bayes import GaussianNB, MultinomialNB
from sklearn.linear_model import LogisticRegression
from eval import eval_func
from sklearn.neighbors import KNeighborsClassifier

from sklearn.grid_search import GridSearchCV
from sklearn.cross_validation import StratifiedKFold
from sklearn.cross_validation import PredefinedSplit
from sklearn import svm

from keras.datasets import reuters
from keras.models import Sequential
from keras.layers import Dense, Dropout, Activation
from keras.utils import np_utils
from keras.preprocessing.text import Tokenizer
from keras.optimizers import SGD

#np.random.seed(1337)  # for reproducibility

def run_test():
    max_words = 1000
    batch_size = 32
    nb_epoch = 5

    print('Loading data...')
    (X_train, y_train), (X_test, y_test) = reuters.load_data(nb_words=max_words, test_split=0.2)
    print(len(X_train), 'train sequences')
    print(len(X_test), 'test sequences')

    #print "X_train: {0} y_train: {1} X_test:{2}  y_test:{3}".format(len(X_train[0]), len(y_train[0]), len(X_test[0]), len(y_test[0]))
    #print "X: {0}".format(X_train[0])
    #print "y: {1}".format(y_train[0])


    nb_classes = np.max(y_train)+1
    print(nb_classes, 'classes')

    print('Vectorizing sequence data...')
    tokenizer = Tokenizer(nb_words=max_words)
    X_train = tokenizer.sequences_to_matrix(X_train, mode='binary')
    X_test = tokenizer.sequences_to_matrix(X_test, mode='binary')

    X_test2 = np.array(X_test, copy=True)

    print('X_train shape:', X_train.shape)
    print('X_test shape:', X_test.shape)

    print('Convert class vector to binary class matrix (for use with categorical_crossentropy)')
    Y_train = np_utils.to_categorical(y_train, nb_classes)
    Y_test = np_utils.to_categorical(y_test, nb_classes)
    print('Y_train shape:', Y_train.shape)
    print('Y_test shape:', Y_test.shape)

    print('Building model...')
    model = Sequential()
    model.add(Dense(512, input_shape=(max_words,)))
    model.add(Activation('relu'))
    model.add(Dropout(0.5))
#    model.add(Dense(512, input_shape=(max_words,)))
#    model.add(Activation('relu'))
#    model.add(Dropout(0.5))
    model.add(Dense(nb_classes))
    model.add(Activation('softmax'))

    model.compile(loss='categorical_crossentropy',
                  optimizer='adam',
                  metrics=['accuracy'])

    history = model.fit(X_train, Y_train,
                        nb_epoch=nb_epoch, batch_size=batch_size,
                        verbose=1, validation_split=0.1)
    score = model.evaluate(X_test, Y_test,
                           batch_size=batch_size, verbose=1)
    print('Test score:', score[0])
    print('Test accuracy:', score[1])

    predictions = model.predict_on_batch(X_test2)
    np.savetxt('pred.csv', predictions, fmt='%f', delimiter=',')
    pred2 = np.argmax(predictions, axis=1)
    np.savetxt('predargmax.csv', pred2, fmt='%i', delimiter=',')
    return pred2


def run_mlp(X_train, y_train, X_test, y_test):

    sample_size = len(X_train[0])
    batch_size = 16
    nb_epoch = 5

    print('Loading data...')
    #(X_train, y_train), (X_test, y_test) = reuters.load_data(nb_words=max_words, test_split=0.2)
    print(len(X_train), 'train sequences')
    print(len(X_test), 'test sequences')

    X_test2 = np.array(X_test, copy=True)

    nb_classes = np.max(y_train)+1
    print(nb_classes, 'classes')

    print('Convert class vector to binary class matrix (for use with categorical_crossentropy)')
    Y_train = np_utils.to_categorical(y_train, nb_classes)
    Y_test = np_utils.to_categorical(y_test, nb_classes)
    print('Y_train shape:', Y_train.shape)
    print('Y_test shape:', Y_test.shape)

    print('Building model...')
    model = Sequential()
    model.add(Dense(96, input_shape=(sample_size,)))
    model.add(Activation('relu'))
    model.add(Dropout(0.5))
    model.add(Dense(nb_classes))
    model.add(Activation('softmax'))

    #sgd = SGD(lr=0.1, decay=1e-6, momentum=0.9, nesterov=True)
    model.compile(loss='categorical_crossentropy',
                  optimizer='adam', #sgd,  #'adam',
                  metrics=['accuracy'])

    history = model.fit(X_train, Y_train,
                        nb_epoch=nb_epoch, batch_size=batch_size,
                        verbose=1, validation_split=0.2)


    score = model.evaluate(X_test, Y_test,
                           batch_size=batch_size, verbose=1)
    print model.metrics_names
    print('Test score:', score[0])
    print('Test accuracy:', score[1])

    predictions = model.predict(X_test)
    np.savetxt('pred.csv', predictions, fmt='%f', delimiter=',')
    pred2 = np.argmax(predictions, axis=1)
    np.savetxt('predargmax.csv', pred2, fmt='%i', delimiter=',')
    return pred2


if __name__ == '__main__':

    #https://www.kaggle.com/c/otto-group-product-classification-challenge/forums/t/14335/1st-place-winner-solution-gilberto-titericz-stanislav-semenov/79599
    run_test()

    print "finished"
    #C:\Users\AlexH\projects\xgb\NetCla\NetCla\data|baseline
    #E:\xgb\NetCla\NetCla\data
    root_folder = "E:/xgb/NetCla/NetCla/data/"
    #root_folder = "C:/Users/ah14aeb/Projects/ECMLComp/data/data/"

    train_file = root_folder + '/train.csv'
    train_targets_file = root_folder + '/train_target.csv'
    validation_file = root_folder + '/valid.csv'
    validation_target_file = root_folder + '/valid_target.csv'

    train_X = np.loadtxt(train_file, skiprows=1, delimiter='\t')
    train_Y = np.loadtxt(train_targets_file, dtype=np.int)
    print("loaded training set")

    valid_X = np.loadtxt(validation_file, skiprows=1, delimiter='\t')
    valid_Y = np.loadtxt(validation_target_file, dtype=np.int)
    print("loaded validation set")

    valid_predictions = run_mlp(train_X, train_Y, valid_X, valid_Y)

    print valid_predictions[0:30]
    print "**"
    print valid_Y[0:30]

    eval_func.eval(valid_Y, valid_predictions)
    print "f1: {0}".format(eval_func.get_f1_macro(valid_Y, valid_predictions))


















"""

model persistence

import keras
from keras.layers import Input, Dense
from keras.models import Model, model_from_json

input = Input(shape=(32,))
output = Dense(1)(input)
model = Model(input=input, output=output)
model.compile(optimizer='adam', loss='mse')

# save the model and weights
with open('model.json', 'w') as fp:
    fp.write(model.to_json())
model.save_weights('model.h5', overwrite=True)

# restore the saved model
with open('model.json', 'r') as fp:
    model_loaded = model_from_json(fp.read())
model_loaded.load_weights('model.h5')

max_words = 1000
batch_size = 32
nb_epoch = 5

print('Loading data...')
(X_train, y_train), (X_test, y_test) = reuters.load_data(nb_words=max_words, test_split=0.2)
print(len(X_train), 'train sequences')
print(len(X_test), 'test sequences')

nb_classes = np.max(y_train)+1
print(nb_classes, 'classes')

print('Vectorizing sequence data...')
tokenizer = Tokenizer(nb_words=max_words)
X_train = tokenizer.sequences_to_matrix(X_train, mode='binary')
X_test = tokenizer.sequences_to_matrix(X_test, mode='binary')
print('X_train shape:', X_train.shape)
print('X_test shape:', X_test.shape)

print('Convert class vector to binary class matrix (for use with categorical_crossentropy)')
Y_train = np_utils.to_categorical(y_train, nb_classes)
Y_test = np_utils.to_categorical(y_test, nb_classes)
print('Y_train shape:', Y_train.shape)
print('Y_test shape:', Y_test.shape)

print('Building model...')
model = Sequential()
model.add(Dense(512, input_shape=(max_words,)))
model.add(Activation('relu'))
model.add(Dropout(0.5))
model.add(Dense(nb_classes))
model.add(Activation('softmax'))

model.compile(loss='categorical_crossentropy',
              optimizer='adam',
              metrics=['accuracy'])

history = model.fit(X_train, Y_train,
                    nb_epoch=nb_epoch, batch_size=batch_size,
                    verbose=1, validation_split=0.1)
score = model.evaluate(X_test, Y_test,
                       batch_size=batch_size, verbose=1)
print('Test score:', score[0])
print('Test accuracy:', score[1])
"""