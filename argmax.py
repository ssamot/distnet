import theano
import theano.tensor as T
from theano.ifelse import ifelse
from keras.utils.theano_utils import alloc_zeros_matrix
import numpy as np



def  argmax2args_step(current_split,current_max,
                         previous_split,previus_max):

        #current_max = current_max.reshape((0,))

        return ifelse(T.gt(current_max, previus_max), (current_split, current_max), (previous_split, previus_max))




def argmax(tensor, maximums):

    [v,score], updates = theano.scan(
                                        fn=argmax2args_step,
                                        sequences=[tensor,maximums],
                                        outputs_info=
                                          [alloc_zeros_matrix(tensor.shape[1]),
                                           theano.shared(np.cast[theano.config.floatX](-9999999.0) )]
                                    )
    return v[-1]



def batchargmax_helper(t, m, p_):
    #return tensor[0:1]
    return argmax(t.dimshuffle(0,1), m.dimshuffle(0))

def batchargmax(tensor, maximums):
     result, updates = theano.scan(
                                        fn=batchargmax_helper,
                                        sequences=[tensor,maximums],
                                        outputs_info=
                                          [alloc_zeros_matrix(tensor.shape[2])]
                                    )

     return result



