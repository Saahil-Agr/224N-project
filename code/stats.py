from __future__ import absolute_import
from __future__ import division

import os
import io
#import json
import sys
#import logging
import re
import csv


#from vocab import get_glove

word_dict = {}
alph_dict = {}
size_dict = {}
alph_dict["cont"] = {}
alph_dict["qn"] = {}
size_dict["cont"] = {}
size_dict["qn"] = {}
size_dict["ans"] = {}
question = ["how","when","which","where","when","what"]
qn_ind = {}

# All inputs are FLAGS in main
def intstr_to_intlist(string):
    """Given a string e.g. '311 9 1334 635 6192 56 639', returns as a list of integers"""
    return [int(s) for s in string.split()]

def split_by_whitespace(sentence):
    words = []
    for space_separated_fragment in sentence.strip().split():
        words.extend(re.split(" ", space_separated_fragment))
    return [w for w in words if w]

def sentence_to_token(sentence, key):
    """Turns an already-tokenized sentence string into word indices
    e.g. "i do n't know" -> [9, 32, 16, 96]
    Note any token that isn't in the word2id mapping gets mapped to the id for UNK
    """
    tokens = split_by_whitespace(sentence) # list of strings
    if key == "qn":
        if tokens[0] in question:
            qn_ind[tokens[0]] = 1 + qn_ind.get(tokens[0],0)
        else:
            qn_ind["other"] = 1 + qn_ind.get("other",0)
    n = len(tokens)
    size_dict[key][n] = 1 + size_dict[key].get(n,0)
    for word in tokens:
        #d[word] = 1 + d.get(word,0)
        for char in word:
            alph_dict[key][char] = 1 + alph_dict[key].get(char,0)

    return tokens

def get_stats(context_path, qn_path, ans_path, context_len, question_len, discard_long):
    """
    This function returns a generator object that yields batches.
    The last batch in the dataset will be a partial batch.
    Read this to understand generators and the yield keyword in Python: https://stackoverflow.com/questions/231767/what-does-the-yield-keyword-do

    Inputs:
      word2id: dictionary mapping word (string) to word id (int)
      context_file, qn_file, ans_file: paths to {train/dev}.{context/question/answer} data files
      batch_size: int. how big to make the batches
      context_len, question_len: max length of context and question respectively
      discard_long: If True, discard any examples that are longer than context_len or question_len.
        If False, truncate those exmaples instead.
    """
    context_file, qn_file, ans_file = open(context_path), open(qn_path), open(ans_path)
    #refill_batches(batches, word2id, context_file, qn_file, ans_file, batch_size, context_len, question_len,
                 #  discard_long)
    #tic = time.time()
    context_line, qn_line, ans_line = context_file.readline(), qn_file.readline(), ans_file.readline() # read the next line from each

    while context_line and qn_line and ans_line: # while you haven't reached the end
        context_tokens = sentence_to_token(context_line, "cont")

        qn_tokens = sentence_to_token(qn_line, "qn")
        ans_span = intstr_to_intlist(ans_line)
        ans_length = ans_span[1] - ans_span[0] + 1
        size_dict["ans"][ans_length] = 1 + size_dict["ans"].get(ans_length,0)
        # read the next line from each file
        context_line, qn_line, ans_line = context_file.readline(), qn_file.readline(), ans_file.readline()


        # read the next line from each file
        context_line, qn_line, ans_line = context_file.readline(), qn_file.readline(), ans_file.readline()
    return True

MAIN_DIR = os.path.relpath(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))) # relative path of the main directory
DEFAULT_DATA_DIR = os.path.join(MAIN_DIR, "data") # relative path of data dir
data_dir = DEFAULT_DATA_DIR
train_context_path = os.path.join(data_dir, "train.context")
train_qn_path = os.path.join(data_dir, "train.question")
train_ans_path = os.path.join(data_dir, "train.span")
dev_context_path = os.path.join(data_dir, "dev.context")
dev_qn_path = os.path.join(data_dir, "dev.question")
dev_ans_path = os.path.join(data_dir, "dev.span")

finished = get_stats(train_context_path,train_qn_path,train_ans_path,400,30,False)
with open('char.csv','wb') as f:
    w = csv.writer(f)
    for key in alph_dict:
        w.writerows(alph_dict[key].items())

with open("len.csv",'wb') as f:
    w  = csv.writer(f)
    for key in size_dict:
            w.writerows(size_dict[key].items())