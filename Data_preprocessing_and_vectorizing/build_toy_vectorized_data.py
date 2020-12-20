"""
Build toy vectorized data useful only to check that classification scripts work normally
"""
import os
import numpy as np

encoded_y_train_fn = os.path.join(os.getcwd(), 'encoded_y_train.npy')
encoded_y_train = np.load(encoded_y_train_fn)

train_doc2vec_fn = os.path.join(os.getcwd(), 'train_doc2vec.npy')
train_doc2vec = np.load(train_doc2vec_fn)

n_samples = 500

encoded_y_toy = encoded_y_train[:n_samples]
encoded_y_toy_fn = os.path.join(os.getcwd(), 'encoded_y_toy.npy')
np.save(encoded_y_toy_fn, encoded_y_toy)

toy_doc2vec = train_doc2vec[:n_samples]
toy_doc2vec_fn = os.path.join(os.getcwd(), 'toy_doc2vec.npy')
np.save(toy_doc2vec_fn, toy_doc2vec)
