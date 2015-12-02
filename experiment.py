import optparse
import sys
sys.setrecursionlimit(100000)

import numpy as np

from babi_helper import get_stories, vectorize_stories, tar, get_supporting_facts
from neuralnetworks import Logic
from config import task_path, tenK_task_path, tasks
from persistance import save_scores
from callbacks import LRScheduler, LearningRateAdapter
from utils import bcolors
from keras.callbacks import EarlyStopping, LearningRateScheduler
import os.path

BATCH_SIZE = 300
EPOCHS = 5000
HOPS = 3
ONLY_SUPPORTING = False

np.set_printoptions(precision=4)
np.set_printoptions(suppress=True)


parser = optparse.OptionParser()
parser.add_option('--onlysup', action="store_true", dest="ONLY_SUPPORTING")
parser.add_option('--10K', action="store_true", dest="big_data")
parser.add_option('--memory', action="store_true", dest="memory")

(opts, args) = parser.parse_args()

if (opts.ONLY_SUPPORTING):
    ONLY_SUPPORTING = True

if (opts.big_data):
    tasks_full_path = [tenK_task_path + task for task in tasks]
else:
    tasks_full_path = [task_path + task for task in tasks]


nn = Logic()


rfilename = "Attention_" + str(HOPS) + "_"
rheader = ""

if(opts.ONLY_SUPPORTING):
    rfilename+="onlysupp"
    rheader+="Only Supporting facts "
else:
    if(opts.memory):
        rfilename+="weak"
        rheader+="Weak Supervision "
    else:
        rfilename+="weak"
        rheader+="Weak Supervision "


if(opts.big_data):
    rfilename+="_10K"
    rheader+="10K "
else:
    rfilename+="_1K"
    rheader+="1K "

if(opts.memory):
    rfilename+="_mem"
    rheader+="Memory"
else:
    rfilename+=""
    rheader+=""



results = [rfilename,str(nn),rheader]

print(bcolors.UNDERLINE + str(results) + bcolors.ENDC)

train = []
test = []

for i, task in enumerate(tasks_full_path):
    if(i> 1):
        break

    print(bcolors.HEADER + "Loading task " + task + " ..." + bcolors.ENDC)
    train_data = tar.extractfile(task.format('train')).readlines()
    train.extend(get_stories(train_data, only_supporting=ONLY_SUPPORTING))

    test_data = tar.extractfile(task.format('test')).readlines()
    test.extend(get_stories(test_data, only_supporting=ONLY_SUPPORTING))



vocab = sorted(reduce(lambda x, y: x | y, (set(story + q + [answer]) for story, q, answer in train + test)))




vocab_size = len(vocab) + 1
word_idx = dict((c, i + 1) for i, c in enumerate(vocab))
story_maxlen = max(map(len, (x for x, _, _ in train + test)))
query_maxlen = max(map(len, (x for _, x, _ in train + test)))

X, Xq, Y = vectorize_stories(train, word_idx, vocab_size, story_maxlen, query_maxlen)
tX, tXq, tY = vectorize_stories(test, word_idx, vocab_size, story_maxlen, query_maxlen)


random_indices = np.random.permutation(X.shape[0])
X = X[random_indices]
Xq = Xq[random_indices]
Y = Y[random_indices]

print('vocab = {}'.format(vocab))
print('X.shape = {}'.format(X.shape))
print('Xq.shape = {}'.format(Xq.shape))
print('Y.shape = {}'.format(Y.shape))
print('story_maxlen, query_maxlen = {}, {}'.format(story_maxlen, query_maxlen))



def multi_predict(mX, mXq, mY, fit):
            pts = []
            for j in range(100):
                pt = fit.predict([mX, mXq], batch_size=BATCH_SIZE).argmax(axis = -1)
                pts.append(pt)
                #pYs.append(tY)
                from scipy.stats import mode
                pts_merged = np.array(pts).transpose((1,0))
                pts_merged = mode(pts_merged, axis = -1)[0]
                pts_merged = pts_merged[:,0]

                acc = np.mean(mY.argmax(axis = -1) == pts_merged)
                print "real_acc", acc, j

            #acc = (xpred.argmax(axis = 1) == mY.argmax(axis = 1)).mean()

            return acc



if(opts.memory):

    np_filename_X = "Xproc_" + str(i)
    np_filename_tX = "tXproc_" + str(i)
    np_filename_ml = "mlproc_" + str(i)
    if(opts.big_data):
        np_filename_X+="_10K.npy"
        np_filename_tX+="_10K.npy"
        np_filename_ml+="_10K.npy"
    else:
        np_filename_X+="_1K.npy"
        np_filename_tX+="_1K.npy"
        np_filename_ml+="1K.npy"



    if(not os.path.exists(np_filename_X)):
        print("Padding...")
        from babi_helper import smartpadding, getmaxsentencelength
           # X = multiplex(X, word_idx)
        max_sentence_length = max(getmaxsentencelength(X, word_idx), getmaxsentencelength(tX, word_idx))
        print("max_sentence_length", max_sentence_length)

        X  = smartpadding(X, word_idx, max_sentence_length)
        tX = smartpadding(tX,  word_idx, max_sentence_length)
        #X = X.astype(np.int32)
        #tX = X.astype(np.int32)

    #     np.save(np_filename_X,X)
    #     np.save(np_filename_tX,tX)
    #     np.save(np_filename_ml,np.array([max_sentence_length]))
    #
    #     print(X.shape)
    # else:
    #     print("Loading saving data..")
    #     X = np.load(np_filename_X)
    #     print X
    #     print X.shape
    #     print type(X)
    #     Xt = np.load(np_filename_tX)
    #     max_sentence_length = np.load(np_filename_ml)[0]
    #     print max_sentence_length

    #
    print("Compiling...")
    attention = nn.distancenet(vocab_size, vocab_size, dropout = True, d_perc = 0.2, hop_depth = HOPS, type = "CCE", maxsize = max_sentence_length)
    #attention = nn.sequencialattention(vocab_size, vocab_size, dropout = True, d_perc = 0.2, hop_depth = 2, type = "CCE", maxsize = max_sentence_length)
    #attention = nn.softmaxattention(vocab_size, vocab_size, dropout = True, d_perc = 0.2, hop_depth = 1, type = "CCE", maxsize = max_sentence_length)

else:
    attention = nn.nomemory(vocab_size, vocab_size, dropout = True, d_perc = 0.1, type = "CCE")


try:
    sch = LRScheduler().schedule
    history = attention.fit([X, Xq], Y, batch_size=BATCH_SIZE, nb_epoch=EPOCHS, show_accuracy=True,
                        callbacks=[LearningRateScheduler(sch)] )
except KeyboardInterrupt:
    print("Stoppping")


for i, task in enumerate(tasks_full_path):


    start = i*1000
    end = (i+1)*1000

    loss, acc = attention.evaluate([tX[start:end], tXq[start:end]], tY[start:end], batch_size=BATCH_SIZE, show_accuracy=True)
    print loss, acc

    # acc = multi_predict(tX[start:end], tXq[start:end], tY[start:end], attention); loss = 0
    # print loss, acc

    print((bcolors.OKGREEN + 'Test loss / test accuracy = {:.5f} / {:.5f}' + bcolors.ENDC).format(loss, acc))
    score = "{:.5f} ".format(acc)

    results.append(score)

    print(bcolors.OKGREEN + str(results) + bcolors.ENDC)
    save_scores(results)


attention.save_weights("./weights/twohops16.weights")