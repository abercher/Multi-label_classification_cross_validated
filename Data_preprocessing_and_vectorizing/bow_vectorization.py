"""
Create bag of word vector representation of the data.
"""
import pandas as pd
import os
from doc2vec_vectorization import clean_text
from sklearn.feature_extraction.text import CountVectorizer
from scipy import sparse
import time
import pickle


def main():

    # Training of the transformer and transformation of train data   # Train and t
    data_train_fn = os.path.join('train_raw.csv')
    df = pd.read_csv(data_train_fn)

    bow_x_train = [clean_text(row["Body"]) for i, row in df.iterrows()]

    bow_vectorizer = CountVectorizer(analyzer='word', ngram_range=(1, 1), binary=True, token_pattern='\S+')

    bow_vectorizer_fn = os.path.join(os.getcwd(), 'bow_vectorizer.pkl')

    start = time.time()
    print(start)
    bow_x_train = bow_vectorizer.fit_transform(bow_x_train)
    stop = time.time()
    print(stop)
    print('Sklearn bow training time {}:'.format(stop - start))

    with open(bow_vectorizer_fn, mode='wb+') as f:
        pickle.dump(bow_vectorizer, f)

    print('Shape vectorized train data:')
    print(bow_x_train.shape)

    vect_x_sparse_fn = os.path.join(os.getcwd(), 'train_bow.npz')
    sparse.save_npz(vect_x_sparse_fn, bow_x_train)

    vect_x_pkl_fn = os.path.join(os.getcwd(), 'train_bow.pkl')
    with open(vect_x_pkl_fn, mode='wb+') as f:
        pickle.dump(bow_x_train, f)

    # Transformation of valid and test data
    data_test_fn = os.path.join('test_raw.csv')
    df = pd.read_csv(data_test_fn)

    bow_x_test = [clean_text(row['Body']) for i, row in df.iterrows()]

    bow_x_test = bow_vectorizer.transform(bow_x_test)
    print('Shape vectorized test data:')
    print(bow_x_test.shape)
    bow_x_test_sparse_fn = os.path.join(os.getcwd(), 'test_bow.npz')
    sparse.save_npz(bow_x_test_sparse_fn, bow_x_test)
    bow_x_test_pkl_fn = os.path.join(os.getcwd(), 'test_bow.pkl')
    with open(bow_x_test_pkl_fn, mode='wb+') as f:
        pickle.dump(bow_x_test, f)

    data_valid_fn = os.path.join('valid_raw.csv')
    df = pd.read_csv(data_valid_fn)

    bow_x_valid = [clean_text(row['Body']) for i, row in df.iterrows()]

    bow_x_valid = bow_vectorizer.transform(bow_x_valid)
    print('Shape vectorized valid data:')
    print(bow_x_valid.shape)
    bow_x_valid_sparse_fn = os.path.join(os.getcwd(), 'valid_bow.npz')
    sparse.save_npz(bow_x_valid_sparse_fn, bow_x_valid)
    bow_x_valid_pkl_fn = os.path.join(os.getcwd(), 'valid_bow.pkl')
    with open(bow_x_valid_pkl_fn, mode='wb+') as f:
        pickle.dump(bow_x_valid, f)


if __name__ == '__main__':
    main()
