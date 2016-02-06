# -*- coding: utf-8 -*-
"""
Created on Fri Jul 24 15:31:26 2015

@author: lhannah
"""

from __future__ import division

import sys
from argparse import ArgumentDefaultsHelpFormatter, ArgumentParser
from collections import defaultdict
from numpy import abs, exp, log, log2, percentile, power, zeros
from pandas import DataFrame, Series
from pandas.io.parsers import read_csv
from scipy.special import gammaln

from IPython import embed


def summarize_topics(filenames, dist, max_phrase_len, min_phrase_count, test, out_filename):
    """
    test = 'MALLET' or 'Turbo Topics' or 'Top Words'
    
    """

    state = read_csv(filenames[0], skiprows=2,
                     usecols=[0, 4, 5], header=0,
                     names=['doc', 'word', 'topic'], sep=' ')
    state['word'] = state['word'].astype(str)

    topics = read_csv(filenames[1], sep='(?: |\t)', engine='python',
                      index_col=0, header=None,
                      names=(['alpha'] + [x for x in xrange(1, 20)]))
    if dist == 'average-posterior':
        topics['prob'] = zeros(len(topics))
        for _, df in state.groupby('doc'):
            topics['prob'] += (topics['alpha'].add(df.groupby('topic').size(),
                                                   fill_value=0) /
                               (topics['alpha'].sum() + len(df)))
        topics['prob'] /= state['doc'].nunique()
    elif dist == 'empirical':
        topics['prob'] = state.groupby('topic')['word'].count() / len(state)
    else:
        topics['prob'] = topics['alpha'] / topics['alpha'].sum()

#    assert topics['prob'].sum() >= 1-1e-15
#    assert topics['prob'].sum() <= 1+1e-15

    num_topics = len(topics)

    phrases = dict()

    print >> sys.stderr, 'Creating candidate n-grams...'

    ngram = []
    prev_doc = -1
    prev_topic = -1
    
    # Assume Turbo Topics is default...
    len_thresh = 0
    if test == 'MALLET':
        len_thresh = 1

    counts = defaultdict(lambda: defaultdict(int))

    for _, row in state.iterrows():
        ngram_temp = row['word']
        ngram_temp = ngram_temp.split()
        ngram_temp = "".join(ngram_temp)
        if row['topic'] == prev_topic and row['doc'] == prev_doc:
            
            if test == 'Top Words':
                ngram = [ngram_temp]
                counts[prev_topic][''.join(ngram)] += 1
            else:
                ngram.append(ngram_temp)
            if test == 'Turbo Topics': # add counts for turbo topics
                if len(ngram) > 1:
                    counts[prev_topic][' '.join(ngram)] += 1
                else:
                    counts[prev_topic][''.join(ngram)] += 1
        else:
            if len(ngram) > len_thresh and len(ngram) <= max_phrase_len:
                if len(ngram) > 1:
                    counts[prev_topic][' '.join(ngram)] += 1
                else:
                    counts[prev_topic][''.join(ngram)] += 1
            if len(ngram) == 2:
                counts[prev_topic][ngram[0]] -= 1
            ngram = [ngram_temp]
            prev_doc = row['doc']
            prev_topic = row['topic']
    # indent or not? the following:
    if len(ngram) > len_thresh and len(ngram) <= max_phrase_len:
        if len(ngram) > 1:
            counts[prev_topic][' '.join(ngram)] += 1
        else:
            counts[prev_topic][''.join(ngram)] += 1
        
    scores = defaultdict(lambda: defaultdict(float))
    total_counts = dict()
    selected_counts_dict = dict()

    for topic in xrange(num_topics):
        n_topic = sum(counts[topic].values())
        for ngram, count in counts[topic].items():
                scores[topic][ngram] = count / n_topic
                if ngram in total_counts.keys():
                    total_counts[ngram] += count
                else:
                    total_counts[ngram] = count
        

    for topic, row in topics.iterrows():
        
        print 'Topic %d: %s' % (topic, ' '.join(row[1:5]))
        print '---'
        print '\n'.join(['%s (%f)' % (x, y) for x, y in
                         sorted(scores[topic].items(), key=(lambda x: x[1]),
                                reverse=True)][:4]) + '\n'
        # Save those items
        for ngram, val in sorted(scores[topic].items(), key = (lambda x : x[1]), reverse = True)[:10]:            
            selected_counts_dict[ngram] = int(total_counts[ngram])
            
        selected_counts_df = DataFrame.from_dict(selected_counts_dict, orient = 'index')
        selected_counts_df.to_csv('selected_phrase_counts_' + out_filename + '_' + '.csv' )
                    

    return


def main():

    p = ArgumentParser(formatter_class=ArgumentDefaultsHelpFormatter)

    p.add_argument('--state', type=str, metavar='<state>', required=True,
                   help='gzipped MALLET state file')
    p.add_argument('--topic-keys', type=str, metavar='<topic-keys>',
                   required=True, help='MALLET topics keys file')
    p.add_argument('--dist', metavar='<dist>', required=True,
                   choices=['average-posterior', 'empirical', 'prior'],
                   help='distribution over topics')
    p.add_argument('--max-phrase-len', type=int, metavar='<max-phrase-len>',
                   default=5, help='maximum phrase length')
    p.add_argument('--min-phrase-count', type=int,
                   metavar='<min-phrase-count>',
                   default=15, help='minimum phrase count')

    args = p.parse_args()

    try:
        summarize_topics([args.state, args.topic_keys], args.dist,
                         args.max_phrase_len, args.min_phrase_count)
    except AssertionError:
        p.print_help()


if __name__ == '__main__':
    main()