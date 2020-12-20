"""
This is a re-implementation of the script found on Kaggle at this address:
https://www.kaggle.com/roccoli/multi-label-classification-with-keras
Since the original script has been written, Keras was updated and the script isn't compatible with
the latest version of Keras. So I needed to rewrite it.
"""
import pandas as pd
import os
import numpy as np
import tensorflow.keras.preprocessing as k_prepr
import json
from keras.models import Sequential
from keras.layers import Dense, Activation, Embedding, GlobalMaxPool1D, Dropout, Conv1D
from keras.callbacks import ReduceLROnPlateau, EarlyStopping, ModelCheckpoint
import pickle
import time

from train_and_evaluate_bin_rel import print_evaluation_scores
from doc2vec_vectorization import clean_text


def pad_pred(pred, n_zeros):
    """
    pred: numpy array containing predictions
    n_zeros: int containing number of columns full of zeros to add on the right of the matrix
    return: numpy array containing padded matrix
    """
    new_pred = np.concatenate((pred, np.zeros((pred.shape[0], n_zeros))), axis=1)
    return new_pred


def main():
    # Seperation into train, test, valid has been done in data_splitting.py
    # Selection of labels to keep has already been done in label_selection.py
    df_train_only_common_fn = os.path.join(os.getcwd(), 'train_raw_only_common_tags.csv')
    df_train = pd.read_csv(df_train_only_common_fn)

    df_test_fn = os.path.join(os.getcwd(), 'test_raw.csv')
    df_test = pd.read_csv(df_test_fn)

    ## Clean text
    df_train['Body'] = df_train['Body'].apply(lambda x: clean_text(x, stem=True))
    df_test['Body'] = df_test['Body'].apply(lambda x: clean_text(x, stem=True))

    ## Load binarized response of training set
    train_y_fn = os.path.join(os.getcwd(), 'encoded_y_train_common.npy')
    train_y = np.load(train_y_fn)

    ## Create and fit keras tokenizer
    n_words = 5000
    tokenizer = k_prepr.text.Tokenizer(
        num_words=n_words,
        filters='!"$%&()*,-./:;<=>?@[\\]^_`{|}~\t\n', # I don't include # and +
        lower=True
    )
    start = time.time()
    tokenizer.fit_on_texts(df_train['Body'])
    stop = time.time()
    fit_time = stop - start
    # Note that tokenizer.word_index is a dictionary mapping ALL words found in the training data to integers,
    # and not only the first num_words most frequent ones.
    print(f"Total number of different words found by the tokenizer = {len(tokenizer.word_index) + 1}")

    ## Vectorize data: words are mapped by tokenizer to their index in the dictionary
    ## which was created during fitting of the tokenizer
    maxlen = 180
    
    def vectorize(text_column):
        indices_sequences = tokenizer.texts_to_sequences(text_column)
        return k_prepr.sequence.pad_sequences(indices_sequences, maxlen=maxlen)

    train_x = vectorize(df_train['Body'])
    print(f"Number of samples in training data: {len(train_x)}")
    print(train_x.shape)
    test_x = vectorize(df_test['Body'])

    ## Load weights computed in another script
    ## Keys of the dictionary are indices of labels and the value is the actual weight of the corresponding label
    com_lbs_idx_2_weights_fn = os.path.join(os.getcwd(), 'com_lbs_idx_2_weights.pkl')
    with open(com_lbs_idx_2_weights_fn, mode='rb') as f:
        weights_di = pickle.load(f)


    emb_size = 100

    ## Load GloVe embeddings and use them as initialization weights
    # Code taken from:
    # https://blog.keras.io/using-pre-trained-word-embeddings-in-a-keras-model.html
    embeddings_index = {}
    embeddings_fn = os.path.join(os.getcwd(), '../glove.6B/glove.6B.100d.txt')
    with open(embeddings_fn, encoding='utf-8', mode='r') as f:
        for line in f:
            values = line.split()
            word = values[0]
            coefs = np.asarray(values[1:], dtype='float32')
            embeddings_index[word] = coefs
    print(f'Found {len(embeddings_index)} word vectors in pretrained GloVe')

    embedding_matrix = np.zeros((n_words, emb_size))
    # As explained above, the tokenizer.word_index contains an index for all the words in the dictionary (not only the
    # ones which will be actually transformed to an index:
    print(embedding_matrix.shape)

    for word, i in tokenizer.word_index.items():
        # In order to keep embeddings only for the most common words
        if i >= n_words:
            continue
        embedding_vector = embeddings_index.get(word)
        if embedding_vector is not None:
            # Words out of GloVe vocabulary are assigned all zeros initial weights
            embedding_matrix[i] = embedding_vector

    ## Creation of the model
    n_filters = 225
    mlb_fn = os.path.join(os.getcwd(), 'mlb_only_com.pkl')
    with open(mlb_fn, mode='rb') as f:
        mlb = pickle.load(f)
    n_classes = len(mlb.classes_)
    print(n_classes)

    model = Sequential()
    model.add(Embedding(n_words, emb_size, input_length=maxlen, weights=[embedding_matrix]))
    #model.add(Embedding(n_words, emb_size, input_length=maxlen))
    model.add(Dropout(0.1))
    model.add(Conv1D(n_filters, 3, padding='valid', activation='relu', strides=1))
    model.add(GlobalMaxPool1D())
    model.add(Dense(n_classes))
    model.add(Activation('sigmoid'))

    model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['categorical_accuracy'])
    model.summary()

    callbacks = [
        ReduceLROnPlateau(),
        #EarlyStopping(patience=4),
        ModelCheckpoint(filepath='model-conv1d.h5', save_best_only=True)
    ]

    start = time.time()
    history = model.fit(train_x,
                        train_y,
                        class_weight=weights_di,
                        epochs=30,
                        batch_size=32,
                        validation_split=0.1,
                        callbacks=callbacks)
    stop = time.time()
    fit_time += stop - start

    print(f'Training time: {fit_time}')

    # Evaluation of the model
    print('Shape test_x:')
    print(test_x.shape)
    test_y_pred = model.predict(test_x)
    print('Shape test_y_pred:')
    print(test_y_pred.shape)
    print('test_y_pred[0][:5]')
    print(test_y_pred[0][:5])
    # One needs to round up the values to have only zeros and ones:
    test_y_pred = np.rint(test_y_pred)

    test_y_fn = os.path.join(os.getcwd(), 'encoded_y_test.npy')
    test_y = np.load(test_y_fn)
    print('Shape of test_y')
    print(test_y.shape)

    # One needs to use zero padding in order to make up for the uncommon
    # labels which were discarded during training:
    n_zeros = test_y.shape[1] - test_y_pred.shape[1]
    test_y_pred_padded = pad_pred(test_y_pred, n_zeros)

    print('Shape of test_y_pred_padded:')
    print(test_y_pred_padded.shape)

    print_evaluation_scores(test_y, test_y_pred_padded)


if __name__ == "__main__":
    main()
