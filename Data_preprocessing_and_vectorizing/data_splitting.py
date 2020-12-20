"""
This is used to create the train and test sets. It uses a lot of code taken from this page:
https://medium.com/towards-artificial-intelligence/multi-label-text-classification-using-scikit-multilearn-case-study-with-stackoverflow-questions-768cb487ad12
"""
import pandas as pd
import os


def main():
    # Merge body and labels tables
    questions_path = os.path.join('statsquestions/Questions.csv')
    questions = pd.read_csv(questions_path, encoding='iso-8859-1')
    tags_path = os.path.join('statsquestions/Tags.csv')
    tags = pd.read_csv(tags_path, encoding='iso-8859-1')
    total_df = pd.merge(questions, tags, on='Id', how='inner')
    concat_tag_df = total_df.groupby(['Id'])['Tag'].apply(','.join).reset_index()
    input_df = pd.merge(questions, concat_tag_df, on='Id', how='inner')[['Title', 'Body', 'Tag']]
    print(input_df.head(5))

    # Split data in train, test, valid

    # Trick to shuffle the data
    input_df = input_df.sample(frac=1, random_state=404).reset_index(drop=True)

    n_rows = len(input_df.index)
    n_train = int(0.8 * n_rows)
    n_test = int(0.1 * n_rows)

    df_train = input_df[:n_train]
    df_test = input_df[n_train: n_train+n_test]
    df_valid = input_df[n_train+n_test:]

    train_fn = os.path.join(os.getcwd(), 'train_raw.csv')
    df_train.to_csv(train_fn)
    test_fn = os.path.join(os.getcwd(), 'test_raw.csv')
    df_test.to_csv(test_fn)
    valid_fn = os.path.join(os.getcwd(), 'valid_raw.csv')
    df_valid.to_csv(valid_fn)


if __name__ == '__main__':
    main()
