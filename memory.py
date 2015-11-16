from babi_helper import supporting_facts_inc, reverse, supporting_facts_inc_old,supporting_facts_inc_all
import numpy as np
from keras.preprocessing.sequence import pad_sequences as pad
from utils import bcolors


def get_supporting_facts_training(X, Xq, word_idx, train_supporting_facts, trained_attention, max_hops=2):
    supporting_sentences = [[0] for i in range(len(X))]
    totalX = []
    totalXq = []
    totalY = []
    enough_memories = [0 for i in range(len(X))]
    allX = X[:]
    selected = [[] for i in range(len(X))]
    leftoversX = []

    for i in range(0, max_hops):
        print (bcolors.BOLD +  "Entering hop " + str(i) + " ..." +  bcolors.ENDC)

        _, combinedXq, leftoverS, supporting_sentences, X, Xq, Y , found_supporting, leftover= supporting_facts_inc(None, Xq, word_idx,
                                                                                                  train_supporting_facts,
                                                                                                  supporting_sentences,
                                                                                                  trained_attention,
                                                                                                  enough_memories, allX, selected,
                                                                                                  )

        totalX.extend(X)
        totalY.extend(Y)
        totalXq.extend(Xq)
        leftoversX.extend(leftover)






        Xq = combinedXq
        train_supporting_facts = leftoverS



        print (bcolors.BOLD +  "Found supporting facts " + str(len(selected)) + " ..." + bcolors.ENDC)
        print(np.sum(enough_memories), "break")

        if (np.sum(enough_memories) == len(enough_memories) or found_supporting == 0 ):
            print (bcolors.BOLD +  "breaking at hop " + str(i) + " ..." )
            break



    X = pad(totalX, maxlen=max(map(len, totalX)))
    Xq = pad(totalXq, maxlen=max(map(len, totalXq)))
    Y = pad(totalY, maxlen=max(map(len, totalY)))
    leftoversX = pad(leftoversX, maxlen=max(map(len, leftoversX)))
    # print (len(supporting_sentences))
    # for sentence in supporting_sentences:
    #     print(sentence)
    supporting_sentences = pad(supporting_sentences, maxlen=max(map(len, supporting_sentences)))

    #X = reverse(X, word_idx)
    #Xq = reverse(Xq, word_idx)


    # import collections
    # y=collections.Counter([tuple(list(x[0]) + list(x[1])) for x in zip(X,Xq)])
    # for yi,v in y.items():
    #             if(v > 1):
    #                 print yi, v



    print('X.shape = {}'.format(X.shape))
    print('Xq.shape = {}'.format(Xq.shape))
    print('Y.shape = {}'.format(Y.shape))
    print('leftover.shape = {}'.format(leftoversX.shape))
    print('supporting_sentences.shape = {}'.format(supporting_sentences.shape))
    print(bcolors.ENDC)
    return X, Xq, Y, supporting_sentences, leftoversX
