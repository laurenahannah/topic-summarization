# -*- coding: utf-8 -*-
"""
Created on Sat Dec  5 23:51:18 2015

@author: lhannah
"""
import os

def summarize_topics_testing(state_filename,topic_filename,max_phrase_len, out_filename):
    # Run summarize_recursive first
    min_phrase_count_list = [1,5,10]
    laplace_vec = [0, .01, .1, 1] # Proportions of total
    
    # Loop over min phrase count
    for min_phrase_count in min_phrase_count_list:
        output_list = summarize_topics(state_filename, bfc , 'none', 'empirical', max_phrase_len,
                     min_phrase_count)
        phrases = output_list[0]
        state = output_list[1]
        longest_phrase = output_list[2]
        n = len(state)
        mu_vec = [n*x for x in laplace_vec]
        counts = segment_state(state,phrases,longest_phrase)
        mu_idx = 0
        for mu_val in mu_vec:
            print(mu_val)
            print(mu_idx)
            output_df = get_scores(counts,longest_phrase,state,phrases,topic_filename,mu=mu_val,MI='N')
            # Save the selected phrases in a csv
            output_df.to_csv('selected_phrases_' + out_filename + '_' + str(min_phrase_count) + '_' + str(mu_idx) + '.csv' )
            # Make a nice pretty data frame with counts
            selected_counts_dict = dict()
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
            # Make that dict into a dataframe for saving
            selected_counts_df = DataFrame.from_dict(selected_counts_dict, orient = 'index')
            selected_counts_df.to_csv('selected_phrase_counts_' + out_filename + '_' + str(min_phrase_count) + '_' + str(mu_idx) + '.csv' )
            mu_idx += 1
          
def read_and_tokenize(path_name, output_name, stopword_list):
    """
    Read and tokenize into a data frame. Save results as .txt. Designed for
    setup where path_name \ class labels \ local file names
    Ex: read_and_tokenize('./20news-bydate-train','state-news.txt','stopwords_list.txt')
    Inputs:
        - path_name:  name of path where the text folders are stored
        - output_name: name of .txt file to be saved
        - stopword_list: name of file with stopwords to be removed
    """
    # Traverse directory structure
    
    # columns: doc source pos typeindex type topic
    # first 3 are junk
    doc_col = [0,0,0]
    pos_col = [0,0,0]
    type_col = [".",".","."]
    topic_col = [-1,-1,-1]
    typeindex_col = [0,0,0]
    
    class_counter = -1
    doc_counter = 0
    word_dict = dict()
    f_stopwords = open(stopword_list)
    stopwords_string = f_stopwords.read()
    stopwords = stopwords_string.split()
    f_stopwords.close()
    # The entries up to 35 need to be replaced with whitespace
    for dirName, subdirList, fileList in os.walk(path_name):
        # Go through any files in a subdirectory
        print(dirName)
        # Only search end folders
        if len(subdirList) == 0:
            #fileList_short = fileList[:min(100,len(fileList))]
            fileList_short = fileList
            for fName in fileList_short:
                # Open the file
                temp_open = open(dirName+'/'+fName)
                # Read it
                temp_string = temp_open.read()
                # Replace symbols with " "
                for char_temp in stopwords[:46]:
                    temp_string = temp_string.replace(char_temp," ")
                # Strings to list 
                temp_string = temp_string.lower()
                temp_list = temp_string.split()
                # Loop through and add items to dataframe
                pos_counter = 0
                for str_temp in temp_list:
                    if str_temp not in stopwords:
                        # See if word is in dictionary
                        doc_col.append(doc_counter)
                        topic_col.append(class_counter)
                        type_col.append(str_temp)
                        pos_col.append(pos_counter)
                        if str_temp in word_dict.keys():
                            typeindex_temp = word_dict[str_temp]
                        else:
                            typeindex_temp = len(word_dict)
                            word_dict[str_temp] = typeindex_temp
                        typeindex_col.append(typeindex_temp)
                        pos_counter += 1
                        
                doc_counter += 1
        if len(fileList) > 0:
            class_counter += 1
    # Make the dataframe
    na_col = ["NA"]*len(pos_col)
    state = DataFrame(data = {'doc' : doc_col, 'source' : na_col, 
                              'pos' : pos_col, 'typeindex' : typeindex_col, 
                              'type' : type_col, 'topic' : topic_col})
    # Rearrange columns
    cols = state.columns.tolist()
    cols.insert(1,cols.pop(2))
    cols.insert(3,cols.pop(5))
    cols.insert(5,cols.pop(4))
    state = state[cols]
    print(state[:20])                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                           
    state.to_csv(output_name, header=None, index=None, sep=' ', mode='a')
    return state
            
def make_topic_txt(state):
    """
    Use a state file from previous computations to make a txt with:
        topic number \t proportion \t top 20 words separated by a space
    """
    topic_values = set(state['topic'])
    num_topics = len(topic_values)
    topic_prob = state.groupby('topic')['word'].count() / len(state)
    word_set = set(state['word'])
    state_by_topic = state.groupby('topic')
    topic_vec = [0]*len(topic_values)
    top_words = [' ']*len(topic_values)
    topic_prob_vec = [0]*len(topic_values)
    counter = 0
    for topic in topic_values:
        topic_vec[counter] = topic
        topic_prob_vec[counter] = topic_prob[topic]
        n_topic = state_by_topic['word'].get_group(topic).count()
        state_temp = state_by_topic.get_group(topic)
        counts_series = state_temp['word'].value_counts()
        counts_dict = counts_series.to_dict()
        counts_df = DataFrame(counts_dict.items(),columns = ['word', 'count'])
        # Order df in descending order
        counts_df = counts_df.sort(columns=['count'],ascending=False)
        # Write the top 20 elements
        str_temp = ' '.join(counts_df['word'][:20])
        top_words[counter] = str_temp
        counter += 1
    # Write the damn thing
    #print([topic_vec, topic_prob_vec, top_words])
    df_for_output = DataFrame(data = {'topic': topic_vec, 'prob' : topic_prob_vec, 'words' : top_words},columns=['topic','prob','words'])
    df_for_output.to_csv('topic-keys-reuters-50-50.txt',sep='\t',header = False,index = False)
        
    
            
        