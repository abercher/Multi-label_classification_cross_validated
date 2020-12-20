"""
Create doc2vec vector representation of data. Most of the code is taken from this tutorial:
https://medium.com/towards-artificial-intelligence/multi-label-text-classification-using-scikit-multilearn-case-study-with-stackoverflow-questions-768cb487ad12
"""
import pandas as pd
import os
from gensim import utils as gs_utils
import gensim.parsing.preprocessing as gsp
from gensim.models.doc2vec import TaggedDocument, Doc2Vec
from sklearn.base import BaseEstimator
from sklearn import utils as skl_utils
from tqdm import tqdm
import multiprocessing
import numpy as np
import time
import pickle

# This part is copy pasted from:
# https://medium.com/towards-artificial-intelligence/multi-label-text-classification-using-scikit-multilearn-case-study-with-stackoverflow-questions-768cb487ad12
# copy paste starts here >>>
filters = [
           gsp.strip_tags,
           gsp.strip_punctuation,
           gsp.strip_multiple_whitespaces,
           gsp.strip_numeric,
           gsp.remove_stopwords,
           gsp.stem_text
          ]




def clean_text(s, stem=True, remove_sw=True):
    if not stem:
        filters_new = [filt for filt in filters if filt != gsp.stem]
    else:
        filter_new = filters
    if remove_sw:
        filters_new = [filt for filt in filters_new if filt != gsp.remove_stopwords]

    s = s.lower()
    s = gs_utils.to_unicode(s)
    for f in filters_new:
        s = f(s)
    return s


# Perform all transformations listed in filters above and create Doc2Vec embeddings from the input text
class Doc2VecTransformer(BaseEstimator):

    def __init__(self, vector_size=100, learning_rate=0.02, epochs=20):
        self.learning_rate = learning_rate
        self.epochs = epochs
        self._model = None
        self.vector_size = vector_size
        self.workers = multiprocessing.cpu_count() - 1

    def fit(self, df_x, df_y=None):
        tagged_x = [TaggedDocument(clean_text(row['Body']).split(), [index]) for index, row in df_x.iterrows()]
        model = Doc2Vec(documents=tagged_x, vector_size=self.vector_size, workers=self.workers)

        for epoch in range(self.epochs):
            model.train(skl_utils.shuffle([x for x in tqdm(tagged_x)]), total_examples=len(tagged_x), epochs=1)
            model.alpha -= self.learning_rate
            model.min_alpha = model.alpha

        self._model = model
        return self

    def transform(self, df_x):
        return np.asmatrix(np.array([self._model.infer_vector(clean_text(row['Body']).split())
                                     for index, row in df_x.iterrows()]))


def main():
    data_train_fn = os.path.join('train_raw.csv')
    df_train = pd.read_csv(data_train_fn)

    transformer = Doc2VecTransformer()

    # Training of the transformer and transformation of train data
    start = time.time()
    transformer.fit(df_train)
    x_doc2vec_train = transformer.transform(df_train)
    stop = time.time()
    print('Total time: {}'.format(stop-start))
    print('Shape transformed training data:')
    print(x_doc2vec_train.shape)
    x_doc2vec_train_fn = os.path.join(os.getcwd(), 'train_doc2vec.npy')
    np.save(x_doc2vec_train_fn, x_doc2vec_train)

    transformer_fn = os.path.join(os.getcwd(), 'doc2vec_transformer.pkl')
    with open(transformer_fn, mode='wb+') as f:
        pickle.dump(transformer, f)

    # Transformation of valid and test data
    data_test_fn = os.path.join('test_raw.csv')
    df_test = pd.read_csv(data_test_fn)
    x_doc2vec_test = transformer.transform(df_test)
    print("Shape transformed test data:")
    print(x_doc2vec_test.shape)
    x_doc2vec_test_fn = os.path.join(os.getcwd(), 'test_doc2vec.npy')
    np.save(x_doc2vec_test_fn, x_doc2vec_test)

    data_valid_fn = os.path.join('valid_raw.csv')
    df_valid = pd.read_csv(data_valid_fn)
    x_doc2vec_valid = transformer.transform(df_valid)
    print('Shape transformed valid data:')
    print(x_doc2vec_valid.shape)
    x_doc2vec_valid_fn = os.path.join(os.getcwd(), 'valid_doc2vec.npy')
    np.save(x_doc2vec_valid_fn, x_doc2vec_valid)


if __name__ == '__main__':
    main()
