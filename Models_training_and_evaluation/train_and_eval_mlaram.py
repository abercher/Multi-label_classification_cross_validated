"""
Train MLARAM classifier from scikit multi-learn and evaluate it on test set.
"""
import os
import numpy as np
from scipy import sparse
import pickle
import time
from train_and_evaluate_bin_rel import print_evaluation_scores
from skmultilearn.adapt import MLARAM
import json


def main():
    print('skmultilearn MLARAM')

    common_tags_fn = os.path.join(os.getcwd(), 'common_tags_li.json')
    with open(common_tags_fn, encoding='utf-8', mode='r') as f:
        common_tags = json.load(f)

    rare_tags_fn = os.path.join(os.getcwd(), 'rare_tags_li.json')
    with open(rare_tags_fn, encoding='utf-8', mode='r') as f:
        rare_tags = json.load(f)

    all_tags = common_tags + rare_tags

    ## Using BOW representation

    # Train classifier
    train_x_fn = os.path.join(os.getcwd(), 'train_bow.npz')
    train_x = sparse.load_npz(train_x_fn)

    train_y_fn = os.path.join(os.getcwd(), 'encoded_y_train_common.npy')
    train_y = np.load(train_y_fn)

    n_samples = 9000

    train_x = train_x[:n_samples]
    train_y = train_y[:n_samples]

    model = MLARAM(threshold=0.05, vigilance=0.95)
    model_fn = os.path.join(os.getcwd(), 'mlaram_model_bow.pkl')
    print('Training MLARAM on BOW starts')
    start = time.time()
    model.fit(train_x, train_y)
    stop = time.time()
    print('Training MLARAM on BOW stops')
    print('Training time: {}'.format(stop - start))

    # Save model
    with open(model_fn, mode='wb+') as f:
        pickle.dump(model, f)

    # Test classifier on test data
    test_x_fn = os.path.join(os.getcwd(), 'test_bow.npz')
    test_x = sparse.load_npz(test_x_fn)

    test_y_fn = os.path.join(os.getcwd(), 'encoded_y_test.npy')
    test_y = np.load(test_y_fn)

    start = time.time()
    test_y_pred = model.predict(test_x)
    stop = time.time()
    print('Testing time: {}'.format(stop - start))

    # One needs to use zero padding in order to make up for the uncommon
    # labels which were discarded during training:

    print('Before zero padding:')

    print('test_y.shape: {}'.format(test_y.shape))
    print('test_y_pred.shape: {}'.format(test_y_pred.shape))
    n_zeros = len(all_tags) - len(common_tags)
    test_y_pred = np.concatenate((test_y_pred, np.zeros((test_y_pred.shape[0], n_zeros))), axis=1)

    print('After zero padding:')

    print('test_y.shape: {}'.format(test_y.shape))
    print('test_y_pred.shape: {}'.format(test_y_pred.shape))

    print('')
    print('Metrics:')

    print_evaluation_scores(test_y, test_y_pred)


    # Doc2Vec embeddings

    print('With Doc2Vec embeddings:')

    # Train classifier
    train_x_fn = os.path.join(os.getcwd(), 'train_doc2vec.npy')
    train_x = np.load(train_x_fn)
    train_x = train_x[:n_samples]

    train_y_fn = os.path.join(os.getcwd(), 'encoded_y_train_common.npy')
    train_y = np.load(train_y_fn)
    train_y = train_y[:n_samples]

    model = MLARAM(threshold=0.05, vigilance=0.95)
    model_fn = os.path.join(os.getcwd(), 'mlaram_model_doc2vec.pkl')
    print('Training MLARAM on doc2vec starts')
    start = time.time()
    model.fit(train_x, train_y)
    stop = time.time()
    print('Training MLARAM on doc2vec stops')
    print('Training time: {}'.format(stop - start))

    # Save model
    with open(model_fn, mode='wb+') as f:
        pickle.dump(model, f)

    # Test classifier on test data
    test_x_fn = os.path.join(os.getcwd(), 'test_doc2vec.npy')
    test_x = np.load(test_x_fn)
    print(test_x.shape)

    test_y_fn = os.path.join(os.getcwd(), 'encoded_y_test.npy')
    test_y = np.load(test_y_fn)

    start = time.time()
    test_y_pred = model.predict(test_x)
    stop = time.time()

    print('Testing time: {}'.format(stop - start))

    # One needs to use zero padding in order to make up for the uncommon
    # labels which were discarded during training:

    print('Before zero padding')

    print('test_y.shape: {}'.format(test_y.shape))
    print('test_y_pred.shape: {}'.format(test_y_pred.shape))
    n_zeros = len(all_tags) - len(common_tags)
    test_y_pred = np.concatenate((test_y_pred, np.zeros((test_y_pred.shape[0], n_zeros))), axis=1)

    print('After zero padding:')

    print('test_y.shape: {}'.format(test_y.shape))
    print('test_y_pred.shape: {}'.format(test_y_pred.shape))

    print_evaluation_scores(test_y, test_y_pred)


if __name__ == '__main__':
    main()
