#!/usr/bin/env python
# coding: utf-8

# In[1]:


import os
import bs4 as bs
import urllib.request  
import re
import nltk
# nltk.download('averaged_perceptron_tagger')
DATASET_LOCATION = "/home/carry/opensoft/OpenSoft-Data/All_FT"
DESTINATION_LOCATION = "/home/carry/opensoft/Opensoft-2019/api"
from knapsack import knapsack
import heapq 
from gensim.summarization.summarizer import summarize,_set_graph_edge_weights,_build_graph,_build_corpus,_clean_text_by_sentences, _build_hasheable_corpus
from gensim.summarization import keywords
from gensim.summarization.pagerank_weighted import pagerank_weighted as _pagerank
from gensim.summarization.commons import build_graph as _build_graph
from gensim.summarization.commons import remove_unreachable_nodes as _remove_unreachable_nodes
case_filenames = [f for f in os.listdir(DATASET_LOCATION) if not f.startswith(".") ]


# In[2]:


MAX_WORDS=100
for i,case_filename in enumerate(case_filenames[0:5]):
    with open('{}/{}'.format(DATASET_LOCATION,case_filename)) as f:
        text = f.read().strip()
        text = text.split("\n",6)[6]
    sentences = _clean_text_by_sentences(text)
    sent_for_nltk = [sent.text for sent in sentences]
    nltk_str = " ".join(sent_for_nltk)
    corpus = _build_corpus(sentences)
    hashable_corpus = _build_hasheable_corpus(corpus)
    graph = _build_graph(hashable_corpus)
    _set_graph_edge_weights(graph)
    _remove_unreachable_nodes(graph)
    pagerank_scores = _pagerank(graph)
    sentences_by_corpus = dict(zip(hashable_corpus, sentences))
    get_sentences = [sentences_by_corpus[tuple(doc)] for doc in hashable_corpus[:-1]]
    get_scores = [pagerank_scores.get(doc) for doc in hashable_corpus[:-1]]
    
    word_frequencies = {}  
    for word in nltk.word_tokenize(nltk_str):  
            if word not in word_frequencies.keys():
                word_frequencies[word] = 1
            else:
                word_frequencies[word] += 1
    maximum_frequncy = max(word_frequencies.values())
    for word in word_frequencies.keys():  
        word_frequencies[word] = (word_frequencies[word]/maximum_frequncy)
    sentence_scores = {}  
    stopped_sentences = []
    sani_sent_list = []
    new_sent = sent_for_nltk[0]
    for j in range(len(sent_for_nltk)-1):
        last_word = new_sent.split(" ")[-1]
        if last_word and last_word[-1] != ".":
            new_sent += "."
        last_word = last_word[:-1]
        if len(last_word) < 4 or "." in last_word or "/" in last_word:
            new_sent += (" " + sent_for_nltk[j+1])
        else:
            sani_sent_list.append(new_sent)
            new_sent = sent_for_nltk[j+1]
    if new_sent.split(" ")[-1][-1] != ".":
        new_sent += "."
    sani_sent_list.append(new_sent)
    for sent in sani_sent_list:
        j=0
        stopped_sent_words = []
        for word in nltk.word_tokenize(sent.lower()):
            j=j+1
            if word in word_frequencies.keys():
                if sent not in sentence_scores.keys():
                    sentence_scores[sent] = word_frequencies[word]
                else:
                    sentence_scores[sent] += word_frequencies[word]              
                stopped_sent_words.append(word)
            stopped_sentences.append(" ".join(stopped_sent_words))
        sentence_scores[sent]=sentence_scores[sent]/j     
    for j,get_score in enumerate(get_scores):
        if get_scores[j] == None:
            get_scores[j]=0
    j=0
    l=0
    final_sentence_scores={} 
    
    for sent in sani_sent_list:
        j=j+l
        l=0
        if sent not in final_sentence_scores.keys():
                final_sentence_scores[sent]=0
        else:
            final_sentence_scores[sent]+=sentence_scores[sent]
        for sentence in get_sentences[j:-1]:
            if sentence.text[-1]!='.':
                sentence.text+='.'
            if sent.endswith(sentence.text):
                final_sentence_scores[sent]+=get_scores[j]
                l=l+1
                break
            final_sentence_scores[sent]+=get_scores[j]
            l=l+1
    
    summary_sentences = heapq.nlargest(30, final_sentence_scores, key=final_sentence_scores.get)
    
    
    size = [len(s.split(" ")) for s in summary_sentences]
    weights = [final_sentence_scores[s]/len(s.split(" ")) for s in summary_sentences]
    sol = knapsack(size, weights).solve(MAX_WORDS)
    max_weight, selected_sizes = sol
    summary = " ".join(summary_sentences[s] for s in selected_sizes)
    words_in_summary = len(summary.split(" "))
    print("\n File : {} \n Summary : {} \n Words : {} \n".format(case_filename,summary,words_in_summary))
    q=open("{}/{}".format(DESTINATION_LOCATION,case_filename),"w")
    q.write(summary)


# In[ ]:




