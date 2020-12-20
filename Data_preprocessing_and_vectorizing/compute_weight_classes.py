"""
Compute weights of individual labels to compensate for unbalenced dataset.
Weights are computed for every label as the total number of labels (instances) divided
by the number of times this label appears in the data. So if there are 3 labels, a, b, and c,
and that label a appears 3 times in the data, label b appears 5 times in the data and label c
appears 7 times in the data, the weights are
w_a: (3+5+7)/3
w_b: (3+5+7)/5
w_c: (3+5+7)/7
"""
import pandas as pd
import os
import pickle
import json


def main():

    # Let's do it first for the case where we use all the labels

    # Load tags in original format
    tags_path = os.path.join(os.getcwd(), 'statsquestions/Tags.csv')
    tag_df = pd.read_csv(tags_path, encoding='iso-8859-1')

    # The .csv file containing the tags has this format:
    # Id,Tag
    # 1,bayesian
    # 1,prior
    # 1,elicitation
    # 2,distributions
    # So we count how many time each tag occur and put the count per tag in new
    # dataframe
    tag_count_df = tag_df.groupby('Tag', sort='count').size().reset_index(name='count')

    # Compute total number of tags (numerator of the formula)
    n_tag_total = len(tag_df)
    print('')
    print(n_tag_total)
    print('')

    # Compute weights using the formula above and some pandas magic
    tag_count_df['weight'] = n_tag_total / tag_count_df['count']

    # Load the multi-label binarizer to have the label names but also the order in
    # which they are used by the multi-label binarizer (the weights are ultimately
    # saved as value of a dictionary where the keys are the indices of the labels
    # and not the labels themselves)
    mlb_fn = os.path.join(os.getcwd(), 'mlb.pkl')
    with open(mlb_fn, mode='rb') as f:
        mlb = pickle.load(f)

    all_labels = mlb.classes_

    # Create dictionary mapping label index to label weight

    all_lbs_idx_2_weights = {}

    for i, label in enumerate(all_labels):
        all_lbs_idx_2_weights[i] = tag_count_df[tag_count_df['Tag'] == label]['weight'].values[0]
        if i < 20:
            print(all_lbs_idx_2_weights[i])

    all_lbs_idx_2_weights_fn = os.path.join(os.getcwd(), 'all_lbs_idx_2_weihts.pkl')
    with open(all_lbs_idx_2_weights_fn, mode='wb+') as f:
        pickle.dump(all_lbs_idx_2_weights, f)


    # Let's do the same but restrict to common labels

    # Load multi-label binarizer for common tags
    mlb_only_com_fn = os.path.join(os.getcwd(), 'mlb_only_com.pkl')
    with open(mlb_only_com_fn, mode='rb') as f:
        mlb_only_com = pickle.load(f)

    common_labels = mlb_only_com.classes_

    # Filter out rare tags
    common_tag_df = tag_df[tag_df['Tag'].isin(common_labels)]

    # Compute total number of tags (numerator of the formula)
    n_common_tag_total = len(common_tag_df)

    print('')
    print(n_common_tag_total)
    print('')

    # Compute how many times each common tag appear
    common_tag_count_df = common_tag_df.groupby('Tag', sort='count').size().reset_index(name='count')

    # Compute weights using the formula above and some pandas magic
    common_tag_count_df['weight'] = n_common_tag_total / common_tag_count_df['count']

    # Create dictionary mapping label index to label weight

    com_lbs_idx_2_weights = {}

    for i, label in enumerate(common_labels):
        com_lbs_idx_2_weights[i] = common_tag_count_df[common_tag_count_df['Tag'] == label]['weight'].values[0]
        if i < 20:
            print(com_lbs_idx_2_weights[i])

    com_lbs_idx_2_weights_fn = os.path.join(os.getcwd(), 'com_lbs_idx_2_weights.pkl')
    with open(com_lbs_idx_2_weights_fn, mode='wb+') as f:
        pickle.dump(com_lbs_idx_2_weights, f)




if __name__ == '__main__':
    main()
