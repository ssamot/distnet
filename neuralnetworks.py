s__author__ = 'ssamot'

from keras.layers.embeddings import Embedding
from keras.layers.core import Activation, Dropout, TimeDistributedDense, Dense, Merge
from keras.models import Sequential


from layers import InputDropout, AttentionMerge, AttentionRecurrent, DownSample1D
from keras.layers.normalization import BatchNormalization
from keras.layers.recurrent import SimpleRNN, GRU, LSTM
from keras.layers.advanced_activations import LeakyReLU, PReLU, ELU
from keras.constraints import MaxNorm
from keras.layers.noise import GaussianNoise
from keras.regularizers import WeightRegularizer
from keras.layers.convolutional import Convolution1D, MaxPooling1D, UpSample1D
from keras import activations
import theano.tensor as T
from utils import bcolors

size = 2**6
attention = size
alpha = 1.0


init_function = "glorot_normal"

def elu(X):
    return ((X + abs(X)) / 2.0) + alpha * (T.exp((X - abs(X)) / 2.0) - 1)

activations.elu = elu

def reg():
    return None
    #return WeightRegularizer(l2 = 0.002, l1 = 0)

def mx():
    return None
    #return MaxNorm(3.0)

def bn():
    #return BatchNormalization(mode = 0)
    return BatchNormalization(mode = 1, momentum=0.9)



class Logic():
    def __init__(self, embed_hidden_size=size, sent_hidden_size=size, query_hidden_size=size,
                 deep_hidden_size=size, RNN=SimpleRNN):
        self.deep_hidden_size = deep_hidden_size

        self.embed_hidden_size = embed_hidden_size
        self.sent_hidden_size = sent_hidden_size
        self.query_hidden_size = query_hidden_size
        self.RNN = RNN




    def _getopt(self):
        from keras.optimizers import adam, rmsprop, sgd, adadelta
        learning_rate = 0.001
        opt = adam(lr=learning_rate, epsilon = 0.001)
        #opt = rmsprop(learning_rate)
        #opt = adadelta()
        #opt = sgd(learning_rate = 0.01, momentum= 0.8, nesterov=True)
        return opt



    def distancenet(self, vocab_size, output_size,  maxsize = 1, hop_depth = -1, dropout = False, d_perc = 1,  type = "CCE"):
        print(bcolors.UNDERLINE + 'Building nn model...' + bcolors.ENDC)


        sentrnn = Sequential()
        emb = Embedding(vocab_size, self.embed_hidden_size, mask_zero=False,W_constraint=mx(), W_regularizer=reg(), init = init_function)
        sentrnn.add(emb)

        sentrnn.add(MaxPooling1D(pool_length=maxsize))
        #sentrnn.add(UpSample1D(length=4))
        #sentrnn.add(GRU( self.query_hidden_size, return_sequences=True,activation = "elu", init = init_function))
        #sentrnn.add(DownSample1D(length=maxsize))
        #sentrnn.add(Dropout(0.6))
        # sentrnn.add(TimeDistributedDense(self.sent_hidden_size, activation = "elu", init = init_function))
        # sentrnn.add(Dropout(0.2))


        qrnn = Sequential()


        emb = Embedding(vocab_size, self.embed_hidden_size, mask_zero=False,W_constraint=mx(), W_regularizer=reg(), init = init_function)
        qrnn.add(emb)

        qrnn.add(SimpleRNN( self.query_hidden_size, return_sequences=False,activation = "elu", init = init_function))
        #qrnn.add(BatchNormalization(mode = 1, momentum=0.9))
        #qrnn.add(Dense)
        #qrnn.add(AttentionRecurrent(self.query_hidden_size))



        init_qa = [sentrnn, qrnn]
        past = []
        for i in range(hop_depth):
            hop = Sequential()
            l_size = self.sent_hidden_size
            hop.add(AttentionMerge(init_qa + past, input_shape = (None, None, l_size), mode = "distance"))
            hop.add(Dropout(0.05))
            hop.add(TimeDistributedDense(self.sent_hidden_size, activation = "elu", init = init_function))
            hop.add(Dropout(0.05))
            hop.add(AttentionRecurrent(self.sent_hidden_size, init = init_function))
            hop.add(Dropout(0.05))
            past.append(hop)


        model = hop
        model.add(bn())





        self._adddepth(model, output_size, dropout, d_perc, softmax = (type == "CCE"))

        if(type == "CCE"):
            model.compile(optimizer=self._getopt(), loss='categorical_crossentropy', class_mode='categorical')
        else:
            model.compile(optimizer=self._getopt(), loss='mse')



        return model






    def nomemory(self, vocab_size, output_size, dropout = True, d_perc = 1,  type = "CCE"):
        print(bcolors.UNDERLINE + 'Building nn model...' + bcolors.ENDC)

        print (dropout, d_perc)
        sentrnn = Sequential()
        #sentrnn.add(InputDropout(0.3, (None, vocab_size) ))
        sentrnn.add(Embedding(vocab_size, self.embed_hidden_size, mask_zero=True))
        sentrnn.add(self.RNN(self.sent_hidden_size,return_sequences=False, activation = "relu"))




        qrnn = Sequential()
        qrnn.add(Embedding(vocab_size, self.embed_hidden_size, mask_zero=True))
        qrnn.add(self.RNN( self.query_hidden_size, return_sequences=False, activation = "relu"))




        model = Sequential()
        model.add(Merge([sentrnn, qrnn], mode='concat'))
        #merge_size = self.sent_hidden_size + self.query_hidden_size
        if(dropout):
            model.add(Dropout(d_perc))
            #model.add(bn())



        model.add(Dense(self.deep_hidden_size))
        model.add(LeakyReLU())
        #model.add(BatchNormalization((self.deep_hidden_size,), mode = 0))
        if(dropout):
            model.add(Dropout(d_perc))
            model.add(bn())


        self._adddepth(model, output_size, dropout, d_perc, softmax = (type == "CCE"))


        if(type == "CCE"):
            model.compile(optimizer=self._getopt(), loss='categorical_crossentropy', class_mode='categorical')
        else:
            model.compile(optimizer=self._getopt(), loss='mse')


        return model


    def _adddepth(self, model, vocab_size, dropout, d_perc, softmax):
        model.add(Dense( self.deep_hidden_size,W_constraint=mx(),  W_regularizer=reg(), init = init_function))
        model.add(ELU())

        if(dropout):
            model.add(bn())
            model.add(Dropout(d_perc))

        model.add(Dense(self.deep_hidden_size,W_constraint=mx(), init = init_function ))
        model.add(ELU())

        if(dropout):
            model.add(bn())
            model.add(Dropout(d_perc))
        # #
        # # #
        # model.add(Dense(self.deep_hidden_size,W_constraint=mx(), init = init_function ))
        # model.add(ELU())
        #
        # if(dropout):
        #     model.add(bn())
        #     model.add(Dropout(d_perc))

        #model.add(Dense(self.deep_hidden_size,W_constraint=mx(), init = init_function ))
        #model.add(ELU())
        #
        # if(dropout):
        #     model.add(bn())
        #     model.add(Dropout(d_perc))






        model.add(Dense( vocab_size,W_constraint=mx(), W_regularizer=reg(), init = init_function))
        if(softmax):
            model.add(Activation('softmax'))






    def __repr__(self):
        metadata = ('RNN / Embed / Sent / Query / Hidden = {}, {}, {}, {}, {}'.format(self.RNN, self.embed_hidden_size,
                                                                                      self.sent_hidden_size,
                                                                                      self.query_hidden_size,
                                                                                      self.deep_hidden_size))
        return metadata
