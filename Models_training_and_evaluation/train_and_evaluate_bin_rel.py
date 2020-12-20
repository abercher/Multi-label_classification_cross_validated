"""
Train and evaluate tagging model.
"""
import os
import numpy as np
from scipy import sparse
from sklearn.metrics import accuracy_score, f1_score, recall_score, hamming_loss, jaccard_score
from sklearn.multiclass import OneVsRestClassifier
from sklearn.linear_model import LogisticRegression
import pickle
import time
import json


def print_evaluation_scores(y_val, predicted):
    print('accuracy = ', accuracy_score(y_true=y_val, y_pred=predicted))
    print('F1 score macro = ', f1_score(y_true=y_val, y_pred=predicted, average='macro'))
    print('F1 score micro = ', f1_score(y_true=y_val, y_pred=predicted, average='micro'))
    print('F1 score weighted = ', f1_score(y_true=y_val, y_pred=predicted, average='weighted'))
    print('Recall score = ', recall_score(y_true=y_val, y_pred=predicted, average='macro'))
    print('Hamming loss =', hamming_loss(y_true=y_val, y_pred=predicted))
    print('Jaccard score =', jaccard_score(y_true=y_val, y_pred=predicted, average='macro'))


def main():
    # scikit-learn binary relevance
    print('scikit-learn binary relevance model (OneVsRestClassifier):')

    common_tags_fn = os.path.join(os.getcwd(), 'common_tags_li.json')
    with open(common_tags_fn, encoding='utf-8', mode='r') as f:
        common_tags = json.load(f)

    rare_tags_fn = os.path.join(os.getcwd(), 'rare_tags_li.json')
    with open(rare_tags_fn, encoding='utf-8', mode='r') as f:
        rare_tags = json.load(f)

    all_tags = common_tags + rare_tags

    # BOW embeddings

    print('With BOW embeddings')

    # Train classifier
    train_x_fn = os.path.join(os.getcwd(), 'train_bow.npz')
    train_x = sparse.load_npz(train_x_fn)
    train_x = train_x

    train_y_fn = os.path.join(os.getcwd(), 'encoded_y_train_common.npy')
    train_y = np.load(train_y_fn)
    train_y = train_y
    print('train_y.shape:')
    print(train_y.shape)

    model = OneVsRestClassifier(estimator=LogisticRegression(max_iter=4000))
    model_fn = os.path.join(os.getcwd(), 'binary_model_bow.pkl')
    start = time.time()
    model.fit(train_x, train_y)
    stop = time.time()
    #with open(model_fn, mode='rb') as f:
    #    model = pickle.load(f)
    print(start)
    print(stop)
    print('Training time: {}'.format(stop-start))

    # Save model
    with open(model_fn, mode='wb+') as f:
        pickle.dump(model, f)

    # Test classifier on test data
    test_x_fn = os.path.join(os.getcwd(), 'test_bow.npz')
    test_x = sparse.load_npz(test_x_fn)
    print(test_x.shape)

    test_y_fn = os.path.join(os.getcwd(), 'encoded_y_test.npy')
    test_y = np.load(test_y_fn)

    test_y_pred = model.predict(test_x)

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
    train_x = train_x

    model = OneVsRestClassifier(estimator=LogisticRegression(max_iter=4000))
    model_fn = os.path.join(os.getcwd(), 'binary_model_doc2vec.pkl')
    start = time.time()
    model.fit(train_x, train_y)
    stop = time.time()
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

    test_y_pred = model.predict(test_x)
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
