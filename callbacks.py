from __future__ import absolute_import
from __future__ import print_function
import sys

import numpy as np

from keras.callbacks import Callback




class LRScheduler():

    def __init__(self):
        self.lr = 0.01
    def schedule(self, epoch):
        self.lr = self.lr*0.99
        self.lr = max(self.lr, 0.005)
        print("LR: ",epoch,self.lr, file=sys.stderr)
        return self.lr




class LearningRateAdapter(Callback):
    def __init__(self, monitor='val_acc', ):
        super(Callback, self).__init__()
        self.monitor = monitor
        self.total_loss = 0
        self.total_batches = 0
        self.hits = 0
        self.min_val_loss = np.inf



    # def on_batch_end(self, batch, logs={}):
    #     self.total_loss +=logs["acc"]
    #     self.total_batches+=1.0

    def on_epoch_end(self, epoch, logs={}):

        val_loss = logs.get(self.monitor)
        val_loss = 1-val_loss


        if(val_loss > self.min_val_loss  ):

            self.hits+=1
            print("hits", self.hits, self.model.optimizer.lr.get_value(), val_loss, self.min_val_loss)
            if(self.hits > 5):
                self.hits = 0
                self.min_val_loss = np.inf
                self.model.optimizer.lr.set_value(np.float32(self.model.optimizer.lr.get_value()/2.0))
                lr = self.model.optimizer.lr.get_value()
                print("Decreasing learning rate to", lr)


                if(lr <  1e-6):
                    print("Learning rate too low, stopping", lr)
                    self.model.stop_training = True
        else:
            self.hits = 0

        self.min_val_loss = min(val_loss, self.min_val_loss)



class ComparativeStopping(Callback):
    def __init__(self,  difference = .1, minimum = 0.001,  verbose=1):
        super(Callback, self).__init__()
        self.difference = difference

        self.verbose = verbose
        self.total_loss = 0
        self.total_batches = 0
        self.minimum = minimum
        self.hits = 0



    def on_batch_end(self, batch, logs={}):
        self.total_loss +=logs["acc"]
        self.total_batches+=1.0

    def on_epoch_end(self, epoch, logs={}):
        loss = self.total_loss/self.total_batches
        var_loss = logs.get('val_acc')
        self.total_loss = 0.0
        self.total_batches = 0.0
        #print(logs)
        #print (var_loss - loss)
        if((var_loss - loss  ) > self.difference):

            print("Epoch %05d: early stopping - overfitting %.5f, %.5f " % (epoch, loss, var_loss))
            self.hits+=1
        else:
            self.hits = 0

        if(self.hits > 5):
            self.model.stop_training = True

        if((loss < self.minimum )):
            self.model.stop_training = True
            print("Epoch %05d: early stopping - goal reached %05d, %05d " % (epoch, loss, var_loss))




class ComparativeStopping2(Callback):
    def __init__(self, monitor='val_acc', difference = 0.1, minimum = 0.9999,  verbose=1):
        super(Callback, self).__init__()

        self.difference = difference
        self.monitor = monitor
        self.verbose = verbose
        self.total_loss = 0
        self.total_batches = 0
        self.minimum = minimum



    def on_batch_end(self, batch, logs={}):
        self.total_loss +=logs["acc"]
        self.total_batches+=1.0

    def on_epoch_end(self, epoch, logs={}):
        loss = self.total_loss/self.total_batches
        var_loss = logs.get(self.monitor)
        self.total_loss = 0.0
        self.total_batches = 0.0
        #print (var_loss - loss)
        if((loss - var_loss) > self.difference):
            self.model.stop_training = True
            print("Epoch %05d: early stopping - overfitting %.5f, %.5f " % (epoch, loss, var_loss))

        if((loss > self.minimum )):
            self.model.stop_training = True
            print("Epoch %05d: early stopping - goal reached %05d, %05d " % (epoch, loss, var_loss))



class ThresholdStopping(Callback):
    def __init__(self, monitor='val_loss', minimum = 0.0002, verbose=1):
        super(Callback, self).__init__()

        self.minimum = minimum
        self.monitor = monitor
        self.verbose = verbose
        self.total_loss = 0
        self.total_batches = 0



    def on_batch_end(self, batch, logs={}):
        self.total_loss +=logs["loss"]
        self.total_batches+=1.0

    def on_epoch_end(self, epoch, logs={}):
        loss = self.total_loss/self.total_batches
        self.total_loss = 0
        self.total_batches = 0
        current = logs.get(self.monitor)
        if current is None:
            current = 0

        if current < self.minimum and loss < self.minimum:
            if self.verbose > 0:
                print("Epoch %05d: early stopping - overfitting %.5f, " %( epoch, loss))
            self.model.stop_training = True


    # def on_epoch_begin(self, epoch, logs={}):
    #     if(epoch < 500 ):
    #         lr = np.float32(0.001)
    #     else:
    #         lr =  np.float32(0.0001)
    #     self.model.optimizer.lr.set_value(lr)


