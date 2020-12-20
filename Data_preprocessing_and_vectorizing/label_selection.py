"""
In this cross-validated data set many labels aren't very well represented. Close to 70% of the labels have less
than 100 samples associated to them, and more than 40% of the labels have less than 25 samples associated to them.
This hints that it would be good to ignore some of the labels removing them from the training set and giving up
the idea of predicting them as their large number represents a computational challenge and the too little samples
associated to them make it very unlikely that any model could predict them accurately.
"""
import os
import pandas as pd
from matplotlib import pyplot as plt
import json


def find_prop_tags(n_samples_per_tag, threshold):
    index = 0
    threshold_reached = False
    for i, n_samples in enumerate(n_samples_per_tag):
        if i == 0:
            if n_samples < threshold:
                threshold_reached = True
                break
        else:
            if n_samples_per_tag[i - 1] >= threshold > n_samples:
                index = i
                threshold_reached = True
                break
    if not threshold_reached:
        index = len(n_samples_per_tag)
    return index / len(n_samples_per_tag) * 100, index


def main():
    questions_path = os.path.join('statsquestions/Questions.csv')
    questions = pd.read_csv(questions_path, encoding='iso-8859-1')
    print('Total number of samples:')
    n_questions = len(questions.index)
    print(n_questions)

    tags_path = os.path.join('statsquestions/Tags.csv')
    tags_df = pd.read_csv(tags_path, encoding='iso-8859-1')

    # The .csv file containing the tags has this format:
    # Id,Tag
    # 1,bayesian
    # 1,prior
    # 1,elicitation
    # 2,distributions
    # With one tag and question Id per line. This makes it easy to count how many samples/questions are associated
    # to each tag:
    n_samples_per_tag = list(tags_df.groupby(['Tag']).count().sort_values('Id', ascending=False)['Id'])

    # Compute percentage of tags which have more than a given number of associated samples/questions.
    threshold = 100
    percent_kept_tags, n_kept_tags = find_prop_tags(
        n_samples_per_tag, threshold)
    print('With a threshold of minimum {} samples per tag, we keep {} tags, which represents {}% of the tags'.format(threshold, n_kept_tags, percent_kept_tags))

    # Compute and plot this percentages for different thresholds
    thresholds = range(0, 200, 10)
    percent_labels = [find_prop_tags(n_samples_per_tag, t)[0] for t in thresholds]

    plt.plot(thresholds, percent_labels)
    plt.xlabel('Threshold min nb samples per tag')
    plt.ylabel('Percentage of kept tags')
    plt.show()

    # Compute list of tags which have less than a given number of samples associated to them
    common_tags = sorted(list(tags_df.groupby(['Tag']).count().loc[tags_df.groupby(['Tag']).count()['Id'] >= threshold].index))

    rare_tags = sorted(list(tags_df.groupby(['Tag']).count().loc[tags_df.groupby(['Tag']).count()['Id'] < threshold].index))

    common_tags_fn = os.path.join(os.getcwd(), 'common_tags_li.json')
    with open(common_tags_fn, encoding='utf-8', mode='w+') as f:
        json.dump(common_tags, f, indent=2)

    rare_tags_fn = os.path.join(os.getcwd(), 'rare_tags_li.json')
    with open(rare_tags_fn, encoding='utf-8', mode='w+') as f:
        json.dump(rare_tags, f, indent=2)

    # The next question we want to answer is the following: for a given threshold, if we remove all tags
    # which have less samples/questions associated to them than the given threshold, how many samples
    # end up having no tag associated to them?
    # Let's first count how many samples are there in total:
    n_samples = tags_df['Id'].nunique()
    print('Total number of samples: {}'.format(n_samples))
    print(f'Total nb of rows of tags_df: {len(tags_df.index)}')
    # Let's count how many samples are there once we remove the rare tags
    common_tags_df = tags_df.loc[tags_df['Tag'].isin(common_tags)]
    n_samples_com_tags = common_tags_df['Id'].nunique()
    print('Number of samples having at least one common tag: {}'.format(n_samples_com_tags))
    print(f'Total nb of rows of common_tags_df: {len(common_tags_df.index)}')

    # Once we have established that we're not loosing too many data point,
    # we want to remove samples/questions which have only rare labels
    # from the train set (but not from test and valid).
    train_fn = os.path.join(os.getcwd(), 'train_raw.csv')
    df_train = pd.read_csv(train_fn)
    print(f'Total nb of rows of df_train: {len(df_train.index)}')
    df_train['Common'] = df_train['Tag'].apply(lambda x: len(set(x).intersection(common_tags)) > 0)
    df_train_only_common = df_train[df_train.Common]
    print(f'Total nb of rows of df_train_only_common: {len(df_train_only_common.index)}')
    df_train_only_common_fn = os.path.join(os.getcwd(), 'train_raw_only_common_tags.csv')
    df_train_only_common.to_csv(df_train_only_common_fn)


if __name__ == '__main__':
    main()
