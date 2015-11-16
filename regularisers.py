import theano
import theano.tensor as T
from theano import shared
import numpy as np


from keras.regularizers import Regularizer



class AttentionRegulariser2(Regularizer):
    def __init__(self, l1=0., l2=0.):
        self.l1 = l1
        self.l2 = l2

    def set_layer(self, layer):
        self.layer = layer

    def __call__(self, loss):
        loss += self.l1 * T.sum(T.mean(abs(self.layer.get_attention()), axis=0))
        loss += self.l2 * T.sum(T.mean(self.layer.get_attention() ** 2, axis=0))
        return loss

    def get_config(self):
        return {"name": self.__class__.__name__,
                "l1": self.l1,
                "l2": self.l2}



class AttentionRegulariser(Regularizer):
    def __init__(self, w = 0.01):
        self.w = w

    def set_layer(self, layer):
        self.layer = layer

    def __call__(self, loss):
        attention = self.layer.get_attention() +0.000001
        attention = attention
        entropy = -T.sum(T.log2(attention) * attention, axis = 1)
        entropy = T.mean(entropy)

        loss+= self.w*entropy
        return loss


    def get_config(self):
        return {"name": self.__class__.__name__,
                "l1": self.l1,
                "l2": self.l2}




attention_var = shared(np.array([
                                [1,0,0,0,0,0,0.0,],
                                [1,0,0,0,0,0,0.0,],
                                [0.3333,0.333,0.33333,0,0,0,0.0,],

]))
class DummLayer():
    def get_attention(self):
        return attention_var

if __name__ == "__main__":

    at = AttentionRegulariser(1)
    at.set_layer(DummLayer())



    fn = theano.function([],at(0),on_unused_input='ignore')


    print fn()