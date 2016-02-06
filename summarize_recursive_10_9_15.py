# -*- coding: utf-8 -*-
"""
Created on Fri Oct  9 13:28:27 2015
To do:
    - for competitors (probability, pmi): how many times does it occur in corpus, how long the phrase is

@author: lhannah
"""


from __future__ import division

import sys
from argparse import ArgumentDefaultsHelpFormatter, ArgumentParser
from collections import defaultdict
from numpy import abs, exp, log, log2, percentile, power, zeros, arange, random, histogram, cumsum, isfinite
from pandas import DataFrame, Series
from pandas.io.parsers import read_csv
from scipy.special import gammaln
import matplotlib.pyplot as plt
import csv
from spyderlib.utils.iofuncs import load_dictionary, save_dictionary

from IPython import embed

#####################################
# From David B-H Stackoverflow

def variablesfilter():
    from spyderlib.widgets.dicteditorutils import globalsfilter
    from spyderlib.plugins.variableexplorer import VariableExplorer
    from spyderlib.baseconfig import get_conf_path, get_supported_types

    data = globals()
    settings = VariableExplorer.get_settings()

    get_supported_types()
    data = globalsfilter(data,                   
                         check_all=True,
                         filters=tuple(get_supported_types()['picklable']),
                         exclude_private=settings['exclude_private'],
                         exclude_uppercase=settings['exclude_uppercase'],
                         exclude_capitalized=settings['exclude_capitalized'],
                         exclude_unsupported=settings['exclude_unsupported'],
                         excluded_names=settings['excluded_names']+['settings','In'])
    return data
    
def saveglobals(filename):
    data = variablesfilter()
    save_dictionary(data,filename)

# SAVE:
#savepath = 'test.spydata'

#saveglobals(savepath)

# LOAD:
#globals().update(load_dictionary(fpath)[0])
#data = load_dictionary(fpath)
#phrases = set()
#for idx in xrange(1,(longest_phrase+1)):
#    print(idx)
#    phrases[idx] = counts[idx]['ngram']

#+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
# From 

def file_type(filename):
    magic_dict = {
    "\x1f\x8b\x08": "gz",
    "\x42\x5a\x68": "bz2",
    "\x50\x4b\x03\x04": "zip"
    }

    max_len = max(len(x) for x in magic_dict)
    
    with open(filename) as f:
        file_start = f.read(max_len)
    for magic, filetype in magic_dict.items():
        if file_start.startswith(magic):
            return filetype
    return "txt"
#++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++

def bfc(a, b, c, d, n, a_plus_b, a_plus_c, alpha, alpha_sum):
    """
    Function for computing Bayes factors conditional on n
    """

    num = (gammaln(n + alpha_sum) +
           4 * gammaln(alpha) +
           gammaln(a_plus_b + 2 * alpha - 1.0) +
           gammaln(c + d + 2 * alpha - 1.0) +
           gammaln(a_plus_c + 2 * alpha - 1.0) +
           gammaln(b + d + 2 * alpha - 1.0) +
           2 * gammaln(alpha_sum - 2.0))
    den = (gammaln(alpha_sum) +
           sum([gammaln(alpha + x) for x in [a, b, c, d]]) +
           2 * gammaln(n + alpha_sum - 2.0) +
           4 * gammaln(2 * alpha - 1.0))

    return exp(num - den)


def bfu(a, b, c, d, n, a_plus_b, a_plus_c, alpha, alpha_sum, beta):
    """
    Function for computing Bayes factors unconditional on n
    """

    num = (log(1.0 + 1.0 / beta) +
           gammaln(n + alpha_sum - 1.0) +
           4 * gammaln(alpha) +
           gammaln(a_plus_b + 2 * alpha - 1.0) +
           gammaln(c + d + 2 * alpha - 1.0) +
           gammaln(a_plus_c + 2 * alpha - 1.0) +
           gammaln(b + d + 2 * alpha - 1.0) +
           2 * gammaln(alpha_sum - 2.0))
    den = (gammaln(alpha_sum - 1.0) +
           sum([gammaln(alpha + x) for x in [a, b, c, d]]) +
           2 * gammaln(n + alpha_sum - 2.0) +
           4 * gammaln(2 * alpha - 1.0))

    return exp(num - den)


def csy(a, b, c, d, n, a_plus_b, a_plus_c):
    """
    Function for computing chi-squared tests with a Yates correction
    """

    num = n * power(abs(a * d - b * c) - n / 2.0, 2)
    den = ((a_plus_b) * (c + d) * (a_plus_c) * (b + d))

    return num / den

def recursive_summary(l,state, phrases, test, selection, dist, 
                      max_phrase_len, min_phrase_count,counts):
    """
    Pseudo code:
	- inputs: state file, phrase list from lower levels
	- outputs: phrase list for level (l) (phrases stored in dict of dicts)
	- structure:
		~ make data frame of: n-gram, prefix, suffix, center, count (all)
		~ iterrows over data frame: a = all, a+b = sum(suffix), a+c = sum(prefix), n = sum 
		~ use testing algorithms as implemented
		~ store accepted phrases in dict
      - THIS IS A SUBROUTINE IN summarize_topics
    """
    topics = DataFrame()
    topics['prob'] = state.groupby('topic')['word'].count() / len(state)    
    # Make ngrams for level l
    
    print >> sys.stderr, 'Selecting %d-gram phrases...' % l
    total_phrases = 0
    phrase_and_score = {}
    if l == 1:
        # If unigrams, there is no previous inputs
        # All unigrams are accepted by default        
        ngram = dict()
        for _, row in state.iterrows():
            if row['word'] in ngram.keys():
                ngram[row['word']] += 1
            else:
                ngram[row['word']] = 1
            
        phrases[l] = set([key for key, val in ngram.items() if val >= min_phrase_count])
        scores =  [None]
        total_phrases = len(phrases[l])
        phrase_and_score = dict()
    else:
        # If bigrams or longer, we need to use previous inputs
        ngram = dict([(l,l * [''])]) # Store ngram entries
        doc = dict([(l,l * [-1])]) # Store documents for each word
        topic = dict([(l,l * [-1])]) # Store topics for each word
        num_topics = len(topics) # Total number of topics
        # Do not need to store counts by topic, just 'all'
        
        for idx, row in state.iterrows():
            if l == 1: # We should not be in this situation
                ngram[l] = [row['word']]
                doc[l] = [row['doc']]
                topic[l] = [row['topic']]
            else: # Bigrams or longer
                # Store ngrams, document numbers, and topic number in a list
                ngram[l] = ngram[l][1:] + [row['word']] # Move one word forward
                doc[l] = doc[l][1:] + [row['doc']]
                topic[l] = topic[l][1:] + [row['topic']]
            
            # See if all in same document
            if len(set(doc[l])) == 1:
                counts[l][tuple(ngram[l])][num_topics + 1] += 1
                # See if all in same topic
                if len(set(topic[l])) == 1:
                    counts[l][tuple(ngram[l])][row['topic']] += 1
                    counts[l][tuple(ngram[l])][num_topics] += 1
                    
        # Use tuple keys to make the data frame ngrams
        # Store:
        #   - ngram, its first n-1 words as 'prefix', last n-1 as 'suffix', first word, last, and center
        #   - counts for each topic under topic numbers
        #   - counts for the number of times in the same topic under 'same'
        #   - topic ignorant counts under 'all'
        ngrams = DataFrame.from_records([[' '.join(x), ' '.join(x[:-1]),
                                              ' '.join(x[1:]), x[0],
                                              x[-1], ' '.join(x[1:-1])] + y.tolist()
                                             for x, y in counts[l].items()],
                                            columns=(['ngram', 'prefix',
                                                      'suffix', 'first', 'last', 'center'] +
                                                     range(num_topics) +
                                                     ['same', 'all']))
        n = ngrams['all'].sum()
        counts[l] = ngrams
        
        # Do significance testing
        if test == bfu or test == bfc:
            # If using Bayes factors, set prior parameters
            alpha = 1.0
            alpha_sum = 4 * alpha
            beta = alpha_sum / n
        
        # Precompute the total number of subphrase occurances
        prefix_cache = ngrams.groupby('prefix')['all'].sum()
        suffix_cache = ngrams.groupby('suffix')['all'].sum()
        center_cache = ngrams.groupby('center')['all'].sum()
        first_cache = ngrams.groupby('first')['all'].sum()
        last_cache = ngrams.groupby('last')['all'].sum()
        # Store resulting scores in a list
        scores = len(ngrams) * [None]
        phrase_and_score = list()
        # See if both prefix and suffix are phrases
        for idx, row in ngrams[ngrams['prefix'].isin(phrases[l-1]) &
                               ngrams['suffix'].isin(phrases[l-1]) &
                               (ngrams['all'] >= min_phrase_count)].iterrows():
                                   
            # Do testing for two contingency tables:
            #   - 'first' and 'suffix'
            #   - 'prefix' and 'last'
            a = row['all']
            # Compute values for 'first' vs 'suffix'
            a_plus_b = suffix_cache[row['suffix']]
            a_plus_c = first_cache[row['first']]
            b = a_plus_b - a
            c = a_plus_c - a
            d = n - a_plus_b - c
            # Update test arguments for first test
            args1 = [a, b, c, d, n, a_plus_b, a_plus_c]
            # Compute values for 'prefix' vs 'last'
            a_plus_b = last_cache[row['last']]
            a_plus_c = prefix_cache[row['prefix']]
            b = a_plus_b - a
            c = a_plus_c - a
            d = n - a_plus_b - c
            # Update test arguments for second test
            args2 = [a, b, c, d, n, a_plus_b, a_plus_c]
            # Update parameters to include prior values if Bayes factors
            if test == bfu:
                args1 += [alpha, alpha_sum, beta]
                args2 += [alpha, alpha_sum, beta]
            elif test == bfc:
                args1 += [alpha, alpha_sum]
                args2 += [alpha, alpha_sum]
           
            # Send error if the wrong number of arguments, otherwise run tests
            # For all tests, total score is the less significant value
            if test == bfu:
                if min(len(args1),len(args2)) < 9:
                    print >> sys.stderr, args1
                    print >> sys.stderr, args2
                scores[idx] = max(test(*args1), test(*args2))
            elif test == bfc:
                if min(len(args1),len(args2)) < 9:
                    print >> sys.stderr, args1
                    print >> sys.stderr, args2
                scores[idx] = max(test(*args1), test(*args2))
            else:
                scores[idx] = min(test(*args1), test(*args2))
    
        ngrams['score'] = scores
        # Make mask based on test thresholds
        if test == bfu or test == bfc:
            keep = ngrams['score'] <= (1.0 / 10)
        else:
            keep = ngrams['score'] > 10.83
        # Can include further selection requirements---default is 'none'
        if selection == 'none':
            phrases[l] = set(ngrams[keep]['ngram'])
        else:
            if l == 2:
                phrases[l] = dict(ngrams[keep].set_index('ngram')['score'])
            else:
                m = 2 if selection == 'bigram' else l-1
                if test == bfu or test == bfc:
                    tmp = set([k for k, v in phrases[m].items()
                               if v <= percentile(sorted(phrases[m].values(),
                                                         reverse=True),
                                                  (1.0 - 1.0 / 2**l) * 100)])
                else:
                    tmp = set([k for k, v in phrases[m].items()
                               if v >= percentile(sorted(phrases[m].values()),
                                                  (1.0 - 1.0 / 2**l) * 100)])
                if selection == 'bigram':
                    keep &= Series([all([' '.join(bigram) in tmp for bigram in
                                         zip(words, words[1:])]) for words in
                                    [ngram.split() for ngram in
                                     ngrams['ngram']]])
                            #phrases: set vs dict?
                    phrases[l] = set(ngrams[keep]['ngram'])
                else:
                    keep &= (ngrams['prefix'].isin(tmp) &
                             ngrams['suffix'].isin(tmp))
                    phrases[l] = dict(ngrams[keep].set_index('ngram')['score'])
        
        # Next line is only keeping ngram and counts
        phrase_and_score = zip(ngrams[keep]['ngram'], ngrams[keep]['score'])
        total_phrases = len(phrases[l])
        ngrams.drop(['prefix', 'suffix', 'center','score'], axis=1, inplace=True)
    
    # If new phrases were added, phrases_added is set to True
    phrases_added = False
    if total_phrases > 0:
        phrases_added = True
        
    
    # Return phrases, scores, and whether new phrases are added; stored in a list
    output_list = list()
    output_list.append(phrases) # First output is phrases
    output_list.append(scores)  # Second output is scores
    output_list.append(phrases_added)   # Is at least one new phrase added
    output_list.append(phrases[l])  # Phrases
    output_list.append(counts)  # Counts
    output_list.append(phrase_and_score) # Phrase and score list for graphics/diagnostics
    return output_list



def summarize_topics(filenames, test, selection, dist, max_phrase_len,
                     min_phrase_count):
    """
    Generate phrase list from topics
    
    Input: 
        - filenames: either local .txt or .gz Mallet output
        - test: bfc (Bayes factor conditional), bfu (BF unconditional), or csy (chi2 Yates)
        - selection: choose 'none'
        - dist: choose 'empirical'
        - max_phrase_len: fail safe maximum phrase length, usually 50
        - min_phrase_count: minimal phrase count for phrase selection, usually 5 or 10
    """
    # Check input file type for gz vs txt:
    if file_type(filenames) == 'gz':
        state = read_csv(filenames, compression='gzip', skiprows=2,
                         usecols=[0, 4, 5], header=0,
                         names=['doc', 'word', 'topic'], sep=' ')
    else:
        # This is for a txt file in a local folder
        state = read_csv(filenames,skiprows = 2, usecols = [0,4,5], header = 0, 
                         names = ['doc','word','topic'], sep = ' ')
    
    state['word'] = state['word'].astype(str)
    
    topics = DataFrame()
    topics['prob'] = state.groupby('topic')['word'].count() / len(state)

    num_topics = len(topics)
    # Store phrases in a dictionary where length of ngram is the key, value is set of ngrams
    phrases = dict()

    print >> sys.stderr, 'Creating candidate n-grams...'
    # Call recursive_summary(l,state, phrases, test, selection, dist, max_phrase_len, min_phrase_count)
    # Also need to check if there are still feasible phrases
    keep_going = True
    l = 1 # phrase length
    longest_phrase = 1
    # Store counts in a dictionary where phrase length is key, value is dict
    # that stores counts for phrases in each specific topic, overall in same 
    # topic, and overall without topic info
    counts = dict([(ell, defaultdict(lambda: zeros(num_topics + 2, dtype=int))) for ell in xrange(1, max_phrase_len + 1)])
    bf_score_dict = dict()
    
    while keep_going:
        # Test to see if we should keep going
        print l
        if (l <= 2):
            keep_going = True
        else: # Test to see if overlap in phrases
            phrase_level = phrases[l-1]
            phrase_df = DataFrame.from_records([[x.rsplit(' ',1)[0], 
                                                   x.split(' ',1)[1]] for x in 
                                                   phrase_level], columns = (['beginning', 'ending']))
            intersect_list = list(set(phrase_df['beginning']) & set(phrase_df['ending']))
            if (len(intersect_list) == 0):
                keep_going = False
                break
            elif (l > max_phrase_len):
                keep_going = False
                break
        # Call recursive_summary using previous inputs
        recursive_output = recursive_summary(l,state, phrases, test, selection, dist, 
                                         max_phrase_len, min_phrase_count,counts)
        phrases = recursive_output[0]
        scores = recursive_output[1]
        new_phrases = recursive_output[3]
        counts = recursive_output[4]
        # Store BF scores according to phrase for plotting/debugging
        phrase_and_score = recursive_output[5]
        for idx in xrange(len(phrase_and_score)):
            phrase_value = phrase_and_score[idx][0]
            phrase_score = phrase_and_score[idx][1]
            bf_score_dict[phrase_value] = phrase_score
        if recursive_output[2]:
            longest_phrase = l
        l = l + 1
    
    output_list = list()
    output_list.append(phrases)
    output_list.append(state)
    output_list.append(longest_phrase)
    output_list.append(bf_score_dict)
    output_list.append(counts)
    
    return output_list
    


def segment_state(state,phrases,longest_phrase):
    
    #========================================================
    # Segment state file by phrases to get counts:
    #   - unaware of topic to get 'all' counts
    #   - aware of topic to get topic counts
    #========================================================
    state['position'] = range(len(state))
    state_temp = state.copy()
    # Add some columns to state_temp
    state_temp['word_all'] = state['word']
    state_temp['word_topic'] = state['word']
    num_topics = len(set(state['topic']))
    #counts = dict([(i, defaultdict(lambda: zeros(num_topics + 2, dtype=int)))
    #               for i in xrange(1, longest_phrase + 1)]):
    counts = dict([(l, defaultdict(lambda: zeros(num_topics + 2, dtype=int)))
                   for l in xrange(1, longest_phrase + 1)])
                       # store phrase counts here; one col per topic, one for all
    ngram_topic = dict([(i, i * ['']) for i in xrange(1, longest_phrase + 1)])
    ngram_all = dict([(i, i * ['']) for i in xrange(1, longest_phrase + 1)])
    doc = dict([(i, i * [-1]) for i in xrange(1, longest_phrase + 1)])
    topic = dict([(i, i * [-1]) for i in xrange(1, longest_phrase + 1)])
    
    print >> sys.stderr, 'The longest phrase is a %d-gram...' % longest_phrase
    # The plan:
    #   for n in l to 1 by -1:
    #       1. make phrase_all n-grams via apply
    #       2. make phrase_topic n-grams via apply
    #       3. make .groupby('phrase_all')
    #       4. for phrase in phrases[n]:
    #           a. add counter to counts['all'][phrase]
    #           b. store position, phrase
    #       5. repeat 3,4 for topics
    #       6. update state
    for l in xrange(longest_phrase,0,-1):
        # Start with longest phrase length and go to shortest
        phrases_temp = phrases[l]
        print >> sys.stderr, 'We are on phrase length %d ...' % l
        remove_index_idx_topic = list() # Indicies to replace with 'remove_this_entry'
        replace_index_idx_topic = list() # Indicies to replace with string value
        replace_index_val_topic = list() # List of values to replace
        remove_index_idx_all = list() # Indicies to replace with 'remove_this_entry'
        replace_index_idx_all = list() # Indicies to replace with string value
        replace_index_val_all = list() # List of values to replace   
        
        for idx, row in state_temp.iterrows():
            # Replace first word in n-gram with entire n-gram
            # Replace subsequent words with 'remove_this_entry'
            if l > 1:
                ngram_all[l] = ngram_all[l][1:] + [row['word_all']]
                ngram_topic[l] = ngram_topic[l][1:] + [row['word_topic']]
                doc[l] = doc[l][1:] + [row['doc']]
                topic[l] = topic[l][1:] + [row['topic']]
            else:
                ngram_all[l] = [row['word_all']]
                ngram_topic[l] = [row['word_topic']]
                doc[l] = [row['doc']]
                topic[l] = [row['topic']]
            # Do check for 'all' first
            ngram_all_temp = ' '.join(ngram_all[l])
            ngram_topic_temp = ' '.join(ngram_topic[l])
            if len(set(doc[l])) == 1: # all in same document
                if ngram_all_temp in phrases_temp: # Hey, we have a phrase!
                    counts[l][tuple(ngram_all[l])][num_topics + 1] += 1
                    if l > 1:
                        replace_index_idx_all.append(idx)
                        replace_index_val_all.append(ngram_all_temp)
                        ngram_all[l][l-1] = 'remove_this_entry'
                        remove_index_idx_all.extend(range(idx-l,idx))
                # Now see if we have a phrase for topic
                if (len(set(topic[l])) == 1):
                    if ngram_topic_temp in phrases_temp: # Hey, we have a phrase!
                        counts[l][tuple(ngram_topic[l])][row['topic']] += 1
                        counts[l][tuple(ngram_topic[l])][num_topics] += 1
                        if l > 1:
                            replace_index_idx_topic.append(idx)
                            replace_index_val_topic.append(ngram_topic_temp)
                            remove_index_idx_topic.extend(range(idx-l,idx))
                            #word_topic_temp[idx - l + 1] = ngram_topic_temp
                            ngram_topic[l][l-1] = 'remove_this_entry'
#                        ngram_topic[l-1] = 'remove_this_entry'
#                        if l > 1:
#                            for m in xrange(0,l-1):
#                                word_topic_temp[idx - m] = 'remove_this_entry'
                                #ngram_topic[m] = 'remove_this_entry' # Prevent overlap
        
        ngrams = DataFrame.from_records([[' '.join(x), ' '.join(x[:-1]),
                                          ' '.join(x[1:]), x[0],
                                          x[-1], ' '.join(x[1:-1])] + y.tolist()
                                         for x, y in counts[l].items()],
                                        columns=(['ngram', 'prefix',
                                                  'suffix', 'first', 'last', 'center'] +
                                                 range(num_topics) +
                                                 ['same', 'all']))
        counts[l] = ngrams
        # Will be faster if we store values to be added then add outside of loop
        #state_temp['word_all'] = word_all_temp
        #state_temp['word_topic'] = word_topic_temp
        state_update_1 = state_temp.select(lambda x: x in remove_index_idx_topic)
        state_update_1.word_topic = 'remove_this_entry'
        state_temp.update(state_update_1)
        
        state_update_2 = state_temp.select(lambda x: x in replace_index_idx_topic)
        state_update_2.word_topic = replace_index_val_topic
        state_temp.update(state_update_2)
        
        state_update_3 = state_temp.select(lambda x: x in remove_index_idx_all)
        state_update_3.word_all = 'remove_this_entry'
        state_temp.update(state_update_3)
        
        state_update_4 = state_temp.select(lambda x: x in replace_index_idx_all)
        state_update_4.word_topic = replace_index_val_all
        state_temp.update(state_update_4)
        
    return counts
            
def get_scores(counts,longest_phrase,state,phrases,topic_filename,mu='N',MI='N'):
    # mu is Laplace smoother; other value is 1/T
    # MI: option to use code to select unigram phrases according to mutual information
    #   - 'Y':   use MI unigram selection
    #   - 'N':   use KALE selection for n-gram phrases
    topics = read_csv(topic_filename, index_col = 0, header = None, sep = '(?: |\t)',
                      engine = 'python', names = (['alpha'] + [x for x in xrange(1, 21)]))
    scores = defaultdict(lambda: defaultdict(float))
    num_topics = len(topics)
    topics['prob'] = state.groupby('topic')['word'].count() / len(state)
    if (mu == 'N'):
        mu_val = 0
    topic_prob_vec = list(topics['prob'])
    if (mu != 'N'):
        if mu == 'Y':
            mu_val = 10000 # Add mu_val observations to fake phrase
        else:
            mu_val = mu
        mu_vec = [x*mu_val for x in topic_prob_vec]
        mu = 'Y'
        print(sum(mu_vec))
    print(mu_val)
    range_val = 2
    if MI == 'N':
        range_val = longest_phrase + 1
        
    for l in xrange(1, range_val):
        #print >> sys.stderr, l
        
        ngrams = counts[l].copy(deep = False) 
        #Since I am using counts as a global, deep copy can cause problems
        if len(ngrams) > 0:
            n = ngrams['same'].sum() # By def, n > 0
            if (mu == 'Y'):
                d_temp = [x for x in ngrams['same'] if x > 0]
                D = len(d_temp) # Size of lexicon for phrase length l
                D_all = len(ngrams)
                T = len(topics['prob'])
                print(ngrams['same'].sum())
                #for col_val in xrange(T):
                #    ngrams[col_val] = ngrams[col_val].add(mu_val)
                ngrams.loc[D_all,:] = ngrams.iloc[0,:].copy(deep = False)
                # Pick a phase name that should not be in the phrase list
                ngrams.loc[D_all,'ngram'] = 'fakety fake fake fake phrase for Laplacing'
                for col_val in xrange(T):
                    ngrams.loc[D_all,5+col_val] = mu_vec[col_val]
                ngrams.loc[D_all,'same'] = mu_val
                print(ngrams['same'].sum())
            n = ngrams['same'].sum()
            ngrams['prob'] = ngrams['same'] / n
        #print >> sys.stderr, len(ngrams)   
        
        if len(ngrams) > 0:
        
            for topic in xrange(num_topics):
    
                n_topic = ngrams[topic].sum()
                p_topic = topics['prob'][topic]
                p_not_topic = 1.0 - p_topic
                
                # Compute KALE values
                for _, row in ngrams[(ngrams['ngram'].isin(phrases[l])) &
                                       (ngrams[topic] > 0)].iterrows():
    
                    p_phrase = row['prob']
                    p_topic_g_phrase = row[topic] / row['same']
                    if (n - row['same'] > 0):
                        p_topic_g_not_phrase = ((n_topic - row[topic]) / (n - row['same']))
                    else:
                        p_topic_g_not_phrase = 0
    
                    p_not_phrase = 1.0 - p_phrase
                    p_not_topic_g_phrase = 1.0 - p_topic_g_phrase
                    p_not_topic_g_not_phrase = 1.0 - p_topic_g_not_phrase
    
                    a = 0.0
    
                    if p_topic_g_phrase != 0.0:
                        a += (p_topic_g_phrase *
                              (log2(p_topic_g_phrase) - log2(p_topic)))
                    if p_not_topic_g_phrase != 0.0:
                        a += (p_not_topic_g_phrase *
                              (log2(p_not_topic_g_phrase) - log2(p_not_topic)))
    
                    b = 0.0
    
                    if p_topic_g_not_phrase != 0.0:
                        b += (p_topic_g_not_phrase *
                              (log2(p_topic_g_not_phrase) - log2(p_topic)))
                    if p_not_topic_g_not_phrase != 0.0:
                        b += (p_not_topic_g_not_phrase *
                              (log2(p_not_topic_g_not_phrase) - log2(p_not_topic)))
    
                    scores[topic][row['ngram']] = p_phrase * a + p_not_phrase * b
                    #if row['all'] < 5:
                    #    scores[topic][row['ngram']] = 0
    print '\n'
    output_df = DataFrame()
    topic_number = list()
    phrase_value = list()
    KALE_score = list()
    
    for topic, row in topics.iterrows():
        phrases_temp_list = list(10 * [''])
        scores_temp_list = list(10 * [-1])
        
        topic_number.append(list(10 * [topic]))
        sorted_temp = sorted(scores[topic].items(), key = (lambda x : x[1]), reverse = True)[:10]
        for idx in xrange(10):
            phrases_temp_list[idx] = sorted_temp[idx][0]
            scores_temp_list[idx] = sorted_temp[idx][1]
        phrase_value.append(phrases_temp_list)
        KALE_score.append(scores_temp_list)
            
        print 'Topic %d: %s' % (topic, ' '.join(row[1:11]))
        print '---'
        print ', '.join(['%s (%f)' % (x, y) for x, y in
                         sorted(scores[topic].items(), key=(lambda x: x[1]),
                                reverse=True)][:10]) + '\n'
    
    output_df['phrase'] = phrase_value
    output_df['topic'] = topic_number
    output_df['score'] = KALE_score

    return output_df
    

    
def make_plot(phrases,bf_score_dict,output_df,state,counts,filename):
    # Make a bloxplot that compares the Bayes Factor scores for all phrases and selected ones
    all_bf_scores = power(bf_score_dict.values(), [-1]*len(bf_score_dict))
    all_bf_scores = [x for x in all_bf_scores if isfinite(x)]
    values_all, base_all = histogram(log(all_bf_scores), bins = 100)
    selected_phrase_dict = dict()
    selected_phrase_high_prob_dict = dict()
    topics = DataFrame()
    topics['prob'] = state.groupby('topic')['word'].count() / len(state)
    # Get alpha = 0.5 quantile for topics
    quantile_value = topics['prob'].quantile(q = 0.5)
    selected_counts_dict = dict()
    selected_counts_high_prob_dict = dict()
    for idx, row in output_df.iterrows():
        # Loop over topics
        if 'phase' in set(output_df.columns.values): 
            # Some saved data has this typo---fixed in refactor on 10/23/15
            keys_temp = row['phase']
        else:
            keys_temp = row['phrase']
        for phrase_temp in keys_temp:
            # Get counts---can have unigrams for counts but not Bayes Factors
            phrase_length = len(phrase_temp.split())
            count_df_temp = counts[phrase_length]
            selected_counts_dict[phrase_temp] = int(count_df_temp['all'][count_df_temp['ngram']==phrase_temp]) 
            if topics['prob'][idx] >= quantile_value:
                selected_counts_high_prob_dict[phrase_temp] = int(count_df_temp['all'][count_df_temp['ngram']==phrase_temp])
            if phrase_temp in bf_score_dict.keys():
                selected_phrase_dict[phrase_temp] = power(bf_score_dict[phrase_temp],-1)
                if topics['prob'][idx] >= quantile_value:
                    selected_phrase_high_prob_dict[phrase_temp] = power(bf_score_dict[phrase_temp],-1)
    selected_scores = selected_phrase_dict.values() 
    selected_scores = [x for x in selected_scores if isfinite(x)]
    selected_scores_high_prob = selected_phrase_high_prob_dict.values()
    selected_scores_high_prob = [x for x in selected_scores_high_prob if isfinite(x)]
    values_selected, base_selected = histogram(log(selected_scores),bins = 100)
    values_selected_high_prob, base_selected_high_prob = histogram(log(selected_scores_high_prob),bins = 100)
    cumulative_all = cumsum(values_all)
    cumulative_all_norm = [x / max(cumulative_all) for x in cumulative_all]
    cumulative_selected = cumsum(values_selected)
    cumulative_selected_norm = [x / max(cumulative_selected) for x in cumulative_selected]
    cumulative_selected_high_prob = cumsum(values_selected_high_prob)
    cumulative_selected_high_prob_norm = [x / max(cumulative_selected_high_prob) for x in cumulative_selected_high_prob]
    data = [all_bf_scores, selected_scores]
    
    # Make a zoom cumsum
    values_all_trunc, base_all_trunc = histogram(log(all_bf_scores), bins = [2, 2.5, 3, 3.5, 4, 4.5, 5])
    values_selected_trunc, base_selected_trunc = histogram(log(selected_scores), bins = [2, 2.5, 3, 3.5, 4, 4.5, 5])    
    values_selected_high_prob_trunc, base_selected_high_prob_trunc = histogram(log(selected_scores_high_prob), bins = [2, 2.5, 3, 3.5, 4, 4.5, 5])    
    cumulative_all_norm_trunc = [x / max(cumulative_all) for x in cumsum(values_all_trunc)]    
    cumulative_selected_norm_trunc = [x / max(cumulative_selected) for x in cumsum(values_selected_trunc)]
    cumulative_selected_high_prob_norm_trunc = [x / max(cumulative_selected_high_prob) for x in cumsum(values_selected_high_prob_trunc)]
    f, axarr = plt.subplots(2)    
    
    axarr[0].plot(base_all[:-1], cumulative_all_norm, c='blue', label = 'All Scores')
    axarr[0].plot(base_selected[:-1], cumulative_selected_norm, c='green', label = 'Selected Scores')
    axarr[0].plot(base_selected_high_prob[:-1],cumulative_selected_high_prob_norm, c = 'red', label = 'High Prob Topic')
    axarr[0].set_title('Cumulative Distribution over Entire Range of log(BF Scores)')
    legend = axarr[0].legend(loc='lower right', shadow=True)
    for label in legend.get_texts():
        label.set_fontsize('large')
    
    axarr[1].plot(base_all_trunc[:-1],cumulative_all_norm_trunc, c = 'blue')
    axarr[1].plot(base_selected_trunc[:-1],cumulative_selected_norm_trunc, c = 'green')
    axarr[1].plot(base_selected_high_prob_trunc[:-1],cumulative_selected_high_prob_norm_trunc, c = 'red')
    axarr[1].set_title('Cumulative Distribution over Zoom of log(BF Scores)')
    #plt.boxplot(data)
    #ax = plt.gca()
    #ax.set_yscale('log')
    f.tight_layout()
    plt.savefig(filename + "_BF.pdf", format = 'pdf')
    plt.show()
    #print(min(log(all_bf_scores)))
    #print(values_all)
    #print(base_all)
    
    # Now do this for selected counts
    counts_all = list()
    for phrase_length in counts.keys():
        counts_temp_df = counts[phrase_length]
        if len(counts_temp_df) > 0:
            counts_all.extend(list(counts_temp_df['all'].values.flatten()))
    counts_all = [x for x in counts_all if x > 0]
    
    selected_counts = selected_counts_dict.values()
    selected_counts = [x for x in selected_counts if x > 0]
    selected_counts_high_prob = selected_counts_high_prob_dict.values()
    selected_counts_high_prob = [x for x in selected_counts_high_prob if x > 0]
    
    values_all, base_all = histogram(log(counts_all), bins = 500)
    values_selected, base_selected = histogram(log(selected_counts),bins = 500)
    values_selected_high_prob, base_selected_high_prob = histogram(log(selected_counts_high_prob),bins = 500)
    cumulative_all = cumsum(values_all)
    cumulative_all_norm = [x / max(cumulative_all) for x in cumulative_all]
    cumulative_selected = cumsum(values_selected)
    cumulative_selected_norm = [x / max(cumulative_selected) for x in cumulative_selected]
    cumulative_selected_high_prob = cumsum(values_selected_high_prob)
    cumulative_selected_high_prob_norm = [x / max(cumulative_selected_high_prob) for x in cumulative_selected_high_prob]
    
    values_all_trunc, base_all_trunc = histogram(log(counts_all), bins = list(arange(0.0,4.1,.1)))
    values_selected_trunc, base_selected_trunc = histogram(log(selected_counts), bins = list(arange(0.0,4.1,.1)))    
    values_selected_high_prob_trunc, base_selected_high_prob_trunc = histogram(log(selected_counts_high_prob), bins = list(arange(0.0,4.1,.1)))    
    cumulative_all_norm_trunc = [x / max(cumulative_all) for x in cumsum(values_all_trunc)]    
    cumulative_selected_norm_trunc = [x / max(cumulative_selected) for x in cumsum(values_selected_trunc)]
    cumulative_selected_high_prob_norm_trunc = [x / max(cumulative_selected_high_prob) for x in cumsum(values_selected_high_prob_trunc)]    
    
    f, axarr = plt.subplots(2)    
    
    axarr[0].plot(base_all[:-1], cumulative_all_norm, c='blue', label = 'All Counts')
    axarr[0].plot(base_selected[:-1], cumulative_selected_norm, c='green', label = 'Selected Counts')
    axarr[0].plot(base_selected_high_prob[:-1],cumulative_selected_high_prob_norm, c = 'red', label = 'High Prob Topic')
    axarr[0].set_title('Cumulative Distribution over Entire Range of log(Counts)')
    legend = axarr[0].legend(loc='lower right', shadow=True)
    for label in legend.get_texts():
        label.set_fontsize('large')
    
    axarr[1].plot(base_all_trunc[:-1],cumulative_all_norm_trunc, c = 'blue')
    axarr[1].plot(base_selected_trunc[:-1],cumulative_selected_norm_trunc, c = 'green')
    axarr[1].plot(base_selected_high_prob_trunc[:-1],cumulative_selected_high_prob_norm_trunc, c = 'red')
    axarr[1].set_title('Cumulative Distribution over Zoom of log(Counts)')
    #plt.boxplot(data)
    #ax = plt.gca()
    #ax.set_yscale('log')
    f.tight_layout()
    plt.savefig(filename + "_Counts.pdf", format = 'pdf')
    plt.show()
    
def make_number_plot(phrases,bf_score_dict,state,counts,longest_phrase,topic_filename):
    # Loop over smoothing values and plot results
    n = len(state)
    mu_vec = [0,n/100,n/10,n]
    #f, axarr = plt.subplots(len(mu_vec))
    #g, axarr_2 = plt.subplots(len(mu_vec))
    
    f, axarr = plt.subplots(3)    
    
    axarr_counter = 0
    
    phrase_lengths_array = [0]*len(mu_vec)
    truncated_phrase_lengths_array = [0]*len(mu_vec)    
    short_phrase_lengths_array = [0]*len(mu_vec)    
    
    for mu_val in mu_vec:
        print(mu_val)
        output_df_temp = get_scores(counts,longest_phrase,state,phrases,topic_filename,mu_val,'N')
        # Get counts for output_df_temp
        phrase_lengths = list()
        for topic_number in output_df_temp['phrase'].keys():
            phrase_list_temp = output_df_temp['phrase'][topic_number]
            for phrase_temp in phrase_list_temp:
                phrase_lengths.append(len(phrase_temp.split()))
        longer_phrases = [x for x in phrase_lengths if x > 4]
        shorter_phrases = [x for x in phrase_lengths if x <= 10]
        #axarr[axarr_counter].hist(phrase_lengths, bins = 30, histtype='stepfilled', label = 'mu = ' + str(mu_val))
        #axarr_2[axarr_counter].hist(longer_phrases, bins = 30, histtype='stepfilled', label = 'mu = ' + str(mu_val))
        phrase_lengths_array[axarr_counter] = phrase_lengths
        truncated_phrase_lengths_array[axarr_counter] = longer_phrases
        short_phrase_lengths_array[axarr_counter] = shorter_phrases        
        axarr_counter += 1
    
    axarr[0].hist(phrase_lengths_array, 30, normed=0, histtype='step', fc = 'none',
                            color=['b', 'c', 'r', 'y'],
                            label=['mu = 0', 'mu = n/100', 'mu = n/10', 'mu = n'])  
    axarr[1].hist(truncated_phrase_lengths_array, 25, normed=0, histtype='step', fc = 'none',
                            color=['b', 'c', 'r', 'y'],
                            label=['Crimson', 'Burlywood', 'Chartreuse', 'Red'])
    axarr[2].hist(short_phrase_lengths_array, 8, normed=0, histtype='step', fc = 'none',
                            color=['b', 'c', 'r', 'y'],
                            label=['mu = 0', 'mu = n/100', 'mu = n/10', 'mu = n'])
    f.tight_layout()
    plt.savefig(topic_filename + "_Lengths_Histogram.pdf", format = 'pdf')
    
    #g.tight_layout()
    #plt.savefig(topic_filename + "_Lengths_Histogram_truncated.pdf", format = 'pdf')    
            

def summarize_topics_wrapper(file_names,test,selection,dist,max_phrase_len,min_phrase_count):
    filename_state = file_names[0]
    filename_topic = file_names[1]
    
    output_phrases = summarize_topics(filename_state, test, selection, dist, max_phrase_len,
                     min_phrase_count)
    phrases = output_phrases[0]
    state = output_phrases[1]
    longest_phrase = output_phrases[2]
    
    counts = segment_state(state,phrases,longest_phrase)
    
    get_scores(counts,longest_phrase,state,phrases,filename_topic)



def main():

    tests = {
        'bayes-conditional': bfc,
        'bayes-unconditional': bfu,
        'chi-squared-yates': csy
    }

    p = ArgumentParser(formatter_class=ArgumentDefaultsHelpFormatter)

    p.add_argument('--state', type=str, metavar='<state>', required=True,
                   help='gzipped MALLET state file')
    p.add_argument('--topic-keys', type=str, metavar='<topic-keys>',
                   required=True, help='MALLET topics keys file')
    p.add_argument('--test', metavar='<test>', required=True,
                   choices=['bayes-conditional', 'bayes-unconditional',
                            'chi-squared-yates'],
                   help='hypothesis test for phrase generation')
    p.add_argument('--selection', metavar='<selection>', required=True,
                   choices=['none', 'bigram', 'n-1-gram'],
                   help='additional selection criterion')
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
        summarize_topics_wrapper([args.state, args.topic_keys], tests[args.test],
                         args.selection, args.dist, args.max_phrase_len,
                         args.min_phrase_count)
    except AssertionError:
        p.print_help()


if __name__ == '__main__':
    main()