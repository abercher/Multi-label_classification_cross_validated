import pandas as pd
import os
from matplotlib import pyplot as plt
import re


def main():
    questions_path = os.path.join('statsquestions/Questions.csv')
    questions = pd.read_csv(questions_path, encoding='iso-8859-1')
    print('Questions.csv')
    print('')
    print('Names of the columns:')
    print(questions.columns.values)
    print('')
    print('Number of questions:')
    n_questions = len(questions.index)
    print(n_questions)
    print('')
    print(len(questions.index))
    print('First sample:')
    print(questions.iloc[0])
    print('')
    print('Body of the first question:')
    print(questions.iloc[0]['Body'])
    print('')


    tags_path = os.path.join('statsquestions/Tags.csv')
    tags = pd.read_csv(tags_path, encoding='iso-8859-1')
    print('Tags.csv')
    print('')
    print('Names of the columns')
    print(tags.columns.values)
    print('')
    print('Number of lines in the tag df:')
    print(len(tags.index))
    print('')
    print('First sample:')
    print(tags.iloc[0])
    print('')
    print('Number of different tags:')
    print(len(tags.groupby(['Tag'])))
    print('')
    print('Number of samples per tag for 20 least frequent tags:')
    print(tags.groupby(['Tag']).count().sort_values('Id').head(20))
    print('')
    print('Number of samples per tag for 20 most frequent tags:')
    print(tags.groupby(['Tag']).count().sort_values('Id').tail(20))
    print('')
    print('Proportion of times most common tag appears:')
    print(tags.groupby(['Tag']).count().sort_values('Id')['Id'][-1]/n_questions)
    print('')

    n_samples_per_tag = list(tags.groupby(['Tag']).count().sort_values('Id', ascending=False)['Id'])

    # We will try to plot what percentage of tags are kept if we discard tags with less than M
    # samples assoicated to them, depending on M, for different values of M.
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
        return index/len(n_samples_per_tag)*100

    threshold = 25

    print('With a threshold of minimum {} samples per tag, we keep {}% of the tags'.format(threshold, find_prop_tags(n_samples_per_tag, threshold)))

    # List of thresholds
    thresholds = range(0, 200, 10)
    percent_labels = [find_prop_tags(n_samples_per_tag, t) for t in thresholds]

    plt.plot(thresholds, percent_labels)
    plt.xlabel('Threshold min nb samples per tag')
    plt.ylabel('Percentage of kept tags')
    plt.show()


    num_bins = 10
    #_, _, _ = plt.hist(n_samples_per_tag[: 50], num_bins, facecolor='blue', alpha=0.5)
    #plt.show()
    num_bins = 20
    #_, _, _ = plt.hist(n_samples_per_tag[-500:], num_bins, facecolor='blue', alpha=0.5)
    #plt.show()


    import statistics
    print('Mean number of samples per tag:')
    print(statistics.mean(n_samples_per_tag))
    print('')
    print('Median number of samples per tag:')
    print(statistics.median(n_samples_per_tag))
    print('')

    print(tags.groupby(['Id'])['Tag'].size().reset_index(name='size').sort_values(['size'], ascending=False))
    n_tags_per_sample = list(tags.groupby(['Id'])['Tag'].size().reset_index(name='size')['size'])
    n_samples_per_m_tags = {n: 0 for n in range(1, 6, 1)}
    for n in n_tags_per_sample:
        n_samples_per_m_tags[n] += 1

    # Compute how many different words are in the dictionary
    diff_words = set()
    dictionnary_di = {}
    for i, row in questions.iterrows():
        new_sent = re.sub('[^a-z]', ' ', row['Body'].lower())
        new_sent_set = set(new_sent.split())
        for word in new_sent_set:
            if word in dictionnary_di:
                dictionnary_di[word] += 1
            else:
                dictionnary_di[word] = 1
        diff_words |= new_sent_set

    dictionnary_di = {k: v for k, v in sorted(dictionnary_di.items(), key=lambda item: item[1], reverse=True)}


    dictionnary_li = [(k, v) for k, v in dictionnary_di.items()]

    for i in range(30):
        print(i)
        print(dictionnary_li[i])
        print('')
    for i in range(4999, 5029):
        print(i)
        print(dictionnary_li[i])
        print('')
    for i in range(9999, 10029):
        print(i)
        print(dictionnary_li[i])
        print('')

    print(f'There are {len(diff_words)} different words in the training data.')


if __name__ == '__main__':
    main()
