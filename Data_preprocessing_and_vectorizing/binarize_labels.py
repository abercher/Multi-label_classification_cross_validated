from sklearn.preprocessing import MultiLabelBinarizer
import os
import pandas as pd
import pickle
import numpy as np
import json


# In order not to simplify the task of the classifier, we want the classifier
# to learn only to recognize the most common tags. But in order to assess the
# impact on the test set of ignoring rarer tags, we need to train two instances
# of MultiLabelBinarizer. The first one is used to transform only common tags present
# in the training data into a binary vector. Its output is used to train the classifier
# which predicts the label. The second one is trained is used to transform ANY tag
# present in the training dataset into a (longer) binary vector. It is then used
# to transform the test and validation responses into binary vectors.
# Hence the binary vectors that they output do not have the same size. In order to
# compare the prediction on the test set (which will be shorter) and the
# ground truth on the test set (which will be longer), one will use zero padding.
# But one has to make sure that on their first columns, every column corresponds
# to the same label in the two instances of MultiLabelBinarizer. This is
# why one specifies the classes when creating these instances.

common_tags_fn = os.path.join(os.getcwd(), 'common_tags_li.json')
with open(common_tags_fn, encoding='utf-8', mode='r') as f:
    common_tags = json.load(f)

rare_tags_fn = os.path.join(os.getcwd(), 'rare_tags_li.json')
with open(rare_tags_fn, encoding='utf-8', mode='r') as f:
    rare_tags = json.load(f)

all_tags = common_tags + rare_tags

# Train Binarizer
data_train_fn = os.path.join('train_raw.csv')
df = pd.read_csv(data_train_fn)

y = []
for index, row in df.iterrows():
    y.append(set(row['Tag'].split(',')))

# MultiLabelBinarizer instance transforming all tags in train set

mlb = MultiLabelBinarizer(classes=all_tags)
encoded_y_train = mlb.fit_transform(y)

mlb_fn = os.path.join(os.getcwd(), 'mlb.pkl')
with open(mlb_fn, mode='wb+') as f:
    pickle.dump(mlb, f)

encoded_y_train_fn = os.path.join(os.getcwd(), 'encoded_y_train.npy')

np.save(encoded_y_train_fn, encoded_y_train)

# MultiLabelBinarizer instance transforming only common tags in train set

mlb_only_com = MultiLabelBinarizer(classes=common_tags)
encoded_y_train_common = mlb_only_com.fit_transform(y)

mlb_only_com_fn = os.path.join(os.getcwd(), 'mlb_only_com.pkl')
with open(mlb_only_com_fn, mode='wb+') as f:
    pickle.dump(mlb_only_com, f)

encoded_y_train_common_fn = os.path.join(os.getcwd(), 'encoded_y_train_common.npy')

np.save(encoded_y_train_common_fn, encoded_y_train_common)

# Sanity Check to make sure that all the columns of the output of mlb_only_com
# correspond to the columns of the output of mlb
v1 = np.zeros((1, len(all_tags)))
v1[0, 0] = 1
pred_tags_1 = mlb.inverse_transform(v1)
v1_com = np.zeros((1, len(common_tags)))
v1_com[0, 0] = 1
pred_tags_com_1 = mlb_only_com.inverse_transform(v1_com)


if pred_tags_1 == pred_tags_com_1:
    print('First sanity check passed!')
else:
    print('Failed sanity check:')
    print(pred_tags_1)
    print('is not')
    print(pred_tags_com_1)

# Transform test
data_test_fn = os.path.join('test_raw.csv')
df = pd.read_csv(data_test_fn)

y = []
for index, row in df.iterrows():
    y.append(set(row['Tag'].split(',')))

encoded_y_test = mlb.transform(y)

encoded_y_test_fn = os.path.join(os.getcwd(), 'encoded_y_test.npy')

np.save(encoded_y_test_fn, encoded_y_test)

# Transform valid
data_valid_fn = os.path.join('valid_raw.csv')
df = pd.read_csv(data_valid_fn)

y = []
for index, row in df.iterrows():
    y.append(set(row['Tag'].split(',')))

encoded_y_valid = mlb.transform(y)

encoded_y_valid_fn = os.path.join(os.getcwd(), 'encoded_y_valid.npy')

np.save(encoded_y_valid_fn, encoded_y_valid)


