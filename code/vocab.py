# Copyright 2018 Stanford University
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""This file contains a function to read the GloVe vectors from file,
and return them as an embedding matrix"""

from __future__ import absolute_import
from __future__ import division

from tqdm import tqdm
import numpy as np

_PAD = b"<pad>"
_UNK = b"<unk>"
_START_VOCAB = [_PAD, _UNK]
PAD_ID = 0
UNK_ID = 1
#Per bidaf we have a start word char,end word char and unknown char
_START = b"<strt>"
_END = b"<end>"
START_ID = 2
END_ID = 3
_CHAR_START_VOCAB = [_PAD, _UNK, _START, _END]


def get_glove(glove_path, glove_dim):
    """Reads from original GloVe .txt file and returns embedding matrix and
    mappings from words to word ids.

    Input:
      glove_path: path to glove.6B.{glove_dim}d.txt
      glove_dim: integer; needs to match the dimension in glove_path

    Returns:
      emb_matrix: Numpy array shape (400002, glove_dim) containing glove embeddings
        (plus PAD and UNK embeddings in first two rows).
        The rows of emb_matrix correspond to the word ids given in word2id and id2word
      word2id: dictionary mapping word (string) to word id (int)
      id2word: dictionary mapping word id (int) to word (string)
    """

    print "Loading GLoVE vectors from file: %s" % glove_path
    vocab_size = int(4e5) # this is the vocab size of the corpus we've downloaded

    emb_matrix = np.zeros((vocab_size + len(_START_VOCAB), glove_dim))
    word2id = {}
    id2word = {}

    random_init = True
    # randomly initialize the special tokens
    if random_init:
        emb_matrix[:len(_START_VOCAB), :] = np.random.randn(len(_START_VOCAB), glove_dim)

    # put start tokens in the dictionaries
    idx = 0
    for word in _START_VOCAB:
        word2id[word] = idx
        id2word[idx] = word
        idx += 1

    # go through glove vecs
    with open(glove_path, 'r') as fh:
        for line in tqdm(fh, total=vocab_size):
            line = line.lstrip().rstrip().split(" ")
            word = line[0]
            vector = list(map(float, line[1:]))
            if glove_dim != len(vector):
                raise Exception("You set --glove_path=%s but --embedding_size=%i. If you set --glove_path yourself then make sure that --embedding_size matches!" % (glove_path, glove_dim))
            emb_matrix[idx, :] = vector
            word2id[word] = idx
            id2word[idx] = word
            idx += 1

    final_vocab_size = vocab_size + len(_START_VOCAB)
    assert len(word2id) == final_vocab_size
    assert len(id2word) == final_vocab_size
    assert idx == final_vocab_size

    return emb_matrix, word2id, id2word

def get_char(char_path, char_dim=0, char_embedding = False):
    """
    :param char_path: path to txt file with characters as the first entry and potentially vector encodings for the char
    :param char_dim: dimensionality of the embedding if included
    :param char_embedding: FLAG to let us know to search for embedding or not
    :return: char2id, id2char, char_emb_matrix with last entry none if char_embedding is false
    """

    print "Loading Character vectors from file: %s" % char_path
    vocab_size = 85 #current number of characters we are supporting
    #TODO evaluate this design decision
    #vocab_size = int(4e5) # this is the vocab size of the corpus we've downloaded
    print(char_embedding)
    if char_embedding:
        char_emb_matrix = np.zeros((vocab_size + len(_START_VOCAB), glove_dim))
    else:
        char_emb_matrix = None
    char2id = {}
    id2char = {}

    random_init = True
    # randomly initialize the special tokens
    if random_init and char_embedding:
        emb_matrix[:len(_START_VOCAB), :] = np.random.randn(len(_START_VOCAB), glove_dim)


    # put start tokens in the dictionaries
    idx = 0
    for char in _CHAR_START_VOCAB:
        char2id[char] = idx
        id2char[idx] = char
        idx += 1

    # go through char vecs
    with open(char_path, 'r') as fh:
        for line in tqdm(fh, total=vocab_size):
            line = line.lstrip().rstrip().split(" ")
            char = line[0]
            if char_embedding is True:
                vector = list(map(float, line[1:]))
                if (char_dim != len(vector)):
                    raise Exception("You set --char_path=%s but --embedding_size=%i. If you set --char_path yourself then make sure that --embedding_size matches!" % (char_path, char_dim))
                emb_matrix[idx, :] = vector
            char2id[char] = idx
            id2char[idx] = char
            idx += 1

    final_vocab_size = vocab_size + len(_CHAR_START_VOCAB)
    assert len(char2id) == final_vocab_size
    assert len(id2char) == final_vocab_size
    assert idx == final_vocab_size

    return char_emb_matrix, char2id, id2char