from __future__ import absolute_import
from __future__ import print_function
from functools import reduce
import re
import tarfile

import numpy as np


from keras.preprocessing.sequence import pad_sequences
from keras.datasets.data_utils import get_file



def tokenize(sent):
    '''Return the tokens of a sentence including punctuation.

    >>> tokenize('Bob dropped the apple. Where is the apple?')
    ['Bob', 'dropped', 'the', 'apple', '.', 'Where', 'is', 'the', 'apple', '?']
    '''
    return [x.strip() for x in re.split('(\W+)?', sent) if x.strip()]


def parse_stories(lines, only_supporting=False):
    '''Parse stories provided in the bAbi tasks format

    If only_supporting is true, only the sentences that support the answer are kept.
    '''
    data = []
    story = []
    for line in lines:
        line = line.decode('utf-8').strip()
        nid, line = line.split(' ', 1)
        nid = int(nid)
        if nid == 1:
            story = []
        if '\t' in line:
            q, a, supporting = line.split('\t')
            q = tokenize(q)
            substory = None
            if only_supporting:
                # Only select the related substory
                supporting = map(int, supporting.split())
                substory = [story[i - 1] for i in supporting]
            else:
                # Provide all the substories
                substory = [x for x in story if x]
            data.append((substory, q, a))
            story.append('')
        else:
            sent = tokenize(line)
            story.append(sent)
    return data


def get_supporting_facts(lines):
    '''Parse stories provided in the bAbi tasks format

    If only_supporting is true, only the sentences that support the answer are kept.
    '''
    data = []

    for line in lines:
        line = line.decode('utf-8').strip()
        nid, line = line.split(' ', 1)
        #print(nid)
        if int(nid)== 1:
            counter = 0
            ck = {}
        ck[int(nid)] = counter
        #p#rint(missed)
        if '\t' in line:
            q, a, supporting = line.split('\t')
            supporting = supporting.split(" ")
            # print(supporting)
            # print(ck)
            supporting = [ck[int(s)] for s in supporting]
            #print(supporting)
            data.append(supporting)
        else:
            counter+=1
    return data






def get_stories(lines, only_supporting=False, max_length=None):
    '''Given a file name, read the file, retrieve the stories, and then convert the sentences into a single story.

    If max_length is supplied, any stories longer than max_length tokens will be discarded.
    '''
    data = parse_stories(lines, only_supporting=only_supporting)
    flatten = lambda data: reduce(lambda x, y: x + y, data)
    data = [(flatten(story), q, answer) for story, q, answer in data if
            not max_length or len(flatten(story)) < max_length]
    return data


def vectorize_stories(data, word_idx, vocab_size, story_maxlen, query_maxlen):
    X = []
    Xq = []
    Y = []
    #import nltk
    # word_list2 = [w.strip() for w in word_list if w.strip() not in nltk.corpus.stopwords.words('english')]
    for story, query, answer in data:
        x = [word_idx[w] for w in story]
        xq = [word_idx[w] for w in query]
        y = np.zeros(vocab_size)
        y[word_idx[answer]] = 1
        X.append(x)
        Xq.append(xq)
        Y.append(y)
    return pad_sequences(X, maxlen=story_maxlen), pad_sequences(Xq, maxlen=query_maxlen), np.array(Y)

def splitter(sep, words):
    outlist = []
    curlist = []
    while words:
        word, words = words[0], words[1:]
        curlist.append(word)
        if word in sep:
            outlist.append(curlist)
            curlist = []
    outlist.append(curlist)
    return outlist

def split_sentences(X, word_idx, story_maxlen):
    values = {}
    for story in X:
        sentences = splitter(word_idx["."], list(story))
        for i, sentence in enumerate(sentences):
            #print (sentence)
            #sq = pad_sequences(sentence, maxlen=story_maxlen)
            if(sentence!=[]):
                v = values.get(i, []) + [sentence]
                values[i] = v

    #print(values)

    for key in values.keys():
        v = values[key]
        v+= [[0]for i in range(0, len(X) - len(v))]
        v = pad_sequences(v, maxlen=story_maxlen)
        values[key] = v
        #print(key, v.shape)

    tbr =([values[k] for k in sorted(values.keys())])
    return tbr




def reverse(X, word_idx):
    newX = []
    max_len = 0
    for i,story in enumerate(X):

        story = story[story > 0]
        max_len = max(max_len, len(story))
        sentences = splitter([word_idx["."], word_idx["?"]], list(story))
        sentences = filter(lambda a: a != [], sentences)
        sentences = sentences[::-1]

        newX.append([x for sublist in sentences for x in sublist])


    return pad_sequences(newX, maxlen =  max_len)


def smartpadding(X, word_idx, max_sentence_length):
    newX = []
    for i,story in enumerate(X):
        sentences = splitter([word_idx["."], word_idx["?"] ], list(story))
        sentences = filter(lambda a: a != [], sentences)
        new_sentence = []
        #print(i)
        #print(story)
        #print(sentences)
        for sentence in sentences:
            if(sentence == []):
                continue
            sentence = np.array(sentence)
            #print (sentence[-1])
            s = sentence[sentence > 0]
            if(max_sentence_length < len(s)):
                print("Maximum sentence length is not maximum, found" , len(s), max_sentence_length)

            s = pad_sequences([s], maxlen=max_sentence_length, padding="pre")[0]
            #print(s.shape)
            #print(s)

            new_sentence.extend(s)

        newX.append(new_sentence)

    print ("maxlen", max(map(len, newX)))
    newX = pad_sequences(newX, maxlen=max(map(len, newX)),padding = "pre")
    print(newX.shape)
    return newX


def getmaxsentencelength(X, word_idx):

    max_sentence = 0
    for i,story in enumerate(X):
        sentences = splitter([word_idx["."],word_idx["?"]], list(story))
        sentences = filter(lambda a: a != [], sentences)



        for sentence in sentences:
            sentence = np.array(sentence)
            #print (sentence[-1])
            s = sentence[sentence > 0]
            max_sentence = max(max_sentence, len(s))
    return max_sentence

path = get_file('babi-tasks-v1-2.tar.gz', origin='http://www.thespermwhale.com/jaseweston/babi/tasks_1-20_v1-2.tar.gz')
tar = tarfile.open(path)
