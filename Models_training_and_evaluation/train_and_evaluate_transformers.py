"""
Reproduce (and copy) approach presented on this page:
https://github.com/abhimishra91/transformers-tutorials/blob/master/transformers_multi_label_classification.ipynb
"""
import numpy as np
import pandas as pd
from sklearn import metrics
import transformers
import torch
from torch.utils.data import Dataset, DataLoader, RandomSampler, SequentialSampler
from transformers import BertTokenizer, BertModel, BertConfig
import os
import pickle
import time

from train_and_evaluate_bin_rel import print_evaluation_scores
from doc2vec_vectorization import clean_text


from torch import cuda
device = 'cuda' if cuda.is_available() else 'cpu'


def main():
    # Seperation into train, test, valid has been done in data_splitting.py
    # Selection of labels to keep has already been done in label_selection.py
    df_train_only_common_fn = os.path.join(os.getcwd(), 'train_raw_only_common_tags.csv')
    df_train = pd.read_csv(df_train_only_common_fn)

    n_toy = 100
    #df_train = df_train.head(n_toy)

    df_test_fn = os.path.join(os.getcwd(), 'test_raw.csv')
    df_test = pd.read_csv(df_test_fn)

    ## Clean text
    df_train['Body'] = df_train['Body'].apply(lambda x: clean_text(x, stem=False, remove_sw=False))
    df_test['Body'] = df_test['Body'].apply(lambda x: clean_text(x, stem=False, remove_sw=False))

    ## Load binarized response
    train_y_fn = os.path.join(os.getcwd(), 'encoded_y_train_common.npy')
    train_y = np.load(train_y_fn)

    test_y_fn = os.path.join(os.getcwd(), 'encoded_y_test.npy')
    test_y = np.load(test_y_fn)

    train_y = train_y[:n_toy]

    ## Variables related to training procedure
    MAX_LEN = 200
    TRAIN_BATCH_SIZE = 8
    VALID_BATCH_SIZE = 4
    EPOCHS = 5
    LEARNING_RATE = 1e-05

    ## Using pretrained Tokenizer (whatever "training" means for a tokenizer...)
    tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')

    ## Custom Dataset class to preprocess (tokenization, masking, etc...) the data
    class CustomDataset(Dataset):

        def __init__(self, body, response, tokenizer, max_len):
            self.tokenizer = tokenizer
            self.body = body
            self.response = response
            self.max_len = max_len

        def __len__(self):
            return len(self.response)

        def __getitem__(self, index):
            body = str(self.body[index])
            body = " ".join(body.split())

            inputs = self.tokenizer.encode_plus(
                body,
                None,
                add_special_tokens=True,
                max_length=self.max_len,
                pad_to_max_length=True,
                return_token_type_ids=True,
                truncation=True
            )
            ids = inputs['input_ids']
            mask = inputs['attention_mask']
            token_type_ids = inputs["token_type_ids"]

            return {
                'ids': torch.tensor(ids, dtype=torch.long),
                'mask': torch.tensor(mask, dtype=torch.long),
                'token_type_ids': torch.tensor(token_type_ids, dtype=torch.long),
                'targets': torch.tensor(self.response[index], dtype=torch.float)
            }


    training_set = CustomDataset(body=df_train.Body, response=train_y, tokenizer=tokenizer, max_len=MAX_LEN)
    testing_set = CustomDataset(body=df_test.Body, response=test_y, tokenizer=tokenizer, max_len=MAX_LEN)

    train_params = {'batch_size': TRAIN_BATCH_SIZE,
                    'shuffle': True,
                    'num_workers': 0
                    }
    test_params = {'batch_size': VALID_BATCH_SIZE,
                   'shuffle': True,
                   'num_workers': 0
                   }

    # Data loaders to feed the preprocessed data to the model, batch by batch
    training_loader = DataLoader(training_set, **train_params)
    testing_loader = DataLoader(testing_set, **test_params)

    # Retrieve number of classes
    #mlb_fn = os.path.join(os.getcwd(), 'mlb_only_com.pkl')
    #with open(mlb_fn, mode='rb') as f:
    #    mlb = pickle.load(f)
    #n_classes = len(mlb.classes_)
    n_classes = train_y.shape[1]

    # Custom Pytorch sublass of torch.nn.Module using pretrained BERT
    class BERTClass(torch.nn.Module):
        def __init__(self):
            super(BERTClass, self).__init__()
            self.l1 = transformers.BertModel.from_pretrained('bert-base-uncased')
            self.l2 = torch.nn.Dropout(0.3)
            self.l3 = torch.nn.Linear(768, n_classes)

        def forward(self, ids, mask, token_type_ids):
            _, output_1 = self.l1(ids, attention_mask=mask, token_type_ids=token_type_ids)
            output_2 = self.l2(output_1)
            output = self.l3(output_2)
            return output

    model = BERTClass()
    model.to(device)

    def loss_fn(outputs, targets):
        return torch.nn.BCEWithLogitsLoss()(outputs, targets)

    optimizer = torch.optim.Adam(params=model.parameters(), lr=LEARNING_RATE)

    def train(epoch):
        model.train()
        for _, data in enumerate(training_loader, 0):
            ids = data['ids'].to(device, dtype=torch.long)
            mask = data['mask'].to(device, dtype=torch.long)
            token_type_ids = data['token_type_ids'].to(device, dtype=torch.long)
            targets = data['targets'].to(device, dtype=torch.float)

            outputs = model(ids, mask, token_type_ids)

            optimizer.zero_grad()
            loss = loss_fn(outputs, targets)
            if _ % 5000 == 0:
                print(f'Epoch: {epoch}, Loss:  {loss.item()}')

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()


    def validation(epoch):
        model.eval()
        fin_targets = []
        fin_outputs = []
        with torch.no_grad():
            for _, data in enumerate(testing_loader, 0):
                ids = data['ids'].to(device, dtype=torch.long)
                mask = data['mask'].to(device, dtype=torch.long)
                token_type_ids = data['token_type_ids'].to(device, dtype=torch.long)
                targets = data['targets'].to(device, dtype=torch.float)
                outputs = model(ids, mask, token_type_ids)
                fin_targets.extend(targets.cpu().detach().numpy().tolist())
                fin_outputs.extend(torch.sigmoid(outputs).cpu().detach().numpy().tolist())
        return fin_outputs, fin_targets

    start_total = time.time()

    for epoch in range(EPOCHS):
        print(f'Training epoch {epoch} begins.')
        start = time.time()
        train(epoch)
        stop = time.time()
        print(f'Training epoch {epoch} done.')
        print(f'Training time for this epoch: {stop-start} seconds')
        print()
        start = time.time()
        outputs, targets = validation(epoch)
        stop = time.time()
        print(f'Prediction time for epoch {epoch}: {stop-start} seconds')

        # Round up probabilities
        outputs = np.array(outputs) >= 0.5
        targets = np.array(targets)

        # One needs to use zero padding in order to make up for the uncommon
        # labels which were discarded during training:
        print('Before zero padding')
        print('targets.shape: {}'.format(targets.shape))
        print('outputs.shape: {}'.format(outputs.shape))
        n_zeros = test_y.shape[1] - train_y.shape[1]
        outputs = np.concatenate((outputs, np.zeros((outputs.shape[0], n_zeros))), axis=1)
        print('After zero padding:')
        print('targets.shape: {}'.format(targets.shape))
        print('outputs.shape: {}'.format(outputs.shape))
        print_evaluation_scores(y_val=targets, predicted=outputs)

    stop_total = time.time()
    print(f'Total time: {stop_total-start_total} seconds')


if __name__ == "__main__":
    main()
