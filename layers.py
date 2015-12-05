from theano.sandbox.rng_mrg import MRG_RandomStreams as RandomStreams
from theano.tensor.shared_randomstreams import RandomStreams as RS
from keras.layers.core import MaskedLayer, initializations, activations
import numpy as np

from keras.layers.recurrent import GRU, Recurrent
from keras.utils.theano_utils import shared_zeros, alloc_zeros_matrix
import theano.tensor as T
import theano
from theano.tensor.basic import cast


class InputDropout(MaskedLayer):
    '''
        Hinton's dropout.
    '''

    def __init__(self, p, input_shape):
        super(InputDropout, self).__init__()
        self.p = np.float32(p)
        self.srng = RandomStreams(seed=np.random.randint(10e6))
        self._input_shape = input_shape

    def get_output(self, train=False):
        X = self.get_input(train)
        if (train):
            retain_prob = 1. - self.p
            X *= self.srng.binomial(X.shape, p=retain_prob, dtype=theano.config.floatX)
        X = cast(X, "int32")
        return X

    def get_config(self):
        return {"name": self.__class__.__name__,
                "p": self.p}


class AttentionMerge(MaskedLayer):
    input_ndim = 3

    def __init__(self, layers, output_dim, input_shape, mode='mul', final=False, concat_axis=-1, len_max_sentence=10):
        ''' Merge the output of a list of layers or containers into a single tensor.
            mode: {'sum', 'mul', 'concat'}
        '''
        if len(layers) < 2:
            raise Exception("Please specify two or more input layers (or containers) to merge")
        self.mode = mode
        self.concat_axis = concat_axis
        self.layers = layers
        self.params = []
        self.regularizers = []
        self.constraints = []
        self.updates = []
        self.split_points = len_max_sentence
        self.output_dim = output_dim
        self.input_ndim = AttentionMerge.input_ndim
        self._input_shape = input_shape
        self.softmax = activations.get("softmax")

        self.srng = RS(seed=np.random.randint(10e6))
        self.final = final

        for l in self.layers:
            params, regs, consts, updates = l.get_params()
            self.regularizers += regs
            self.updates += updates
            # params and constraints have the same size
            for p, c in zip(params, consts):
                if p not in self.params:
                    self.params.append(p)
                    self.constraints.append(c)

        self.activation = activations.relu

        self.mask = None

    def get_output_mask(self, train=None):
        return self.mask

    def get_params(self):
        return self.params, self.regularizers, self.constraints, self.updates

    def get_output(self, train=False):
        # can only merge two layers
        memory = self.layers[0].get_output(train)
        # mask = self.layers[0].layers[-1].get_output_mask(train)
        #
        # mask =  T.le(mask, 0.5)
        # last_full_mask = mask.nonzero()[1][0]
        #
        # memory = memory[:,last_full_mask:, :]


        splits = memory.dimshuffle(1, 0, 2)

        # k = self.split_points
        # idx = theano.tensor.arange(memory.shape[0]-1,-1,-k)[::-1]
        # # if(train):
        # #     #k = self.srng.random_integers(10, 20, 30, ndim = 1)[0]
        # #     k = 2
        # #     #idx = theano.tensor.arange(memory.shape[1], k)
        # #     idx = self.srng.shuffle_row_elements(idx)[:k]
        # #     idx = idx.sort()
        # #     self.mask = mask.T[idx].T
        # # else:
        # #     self.mask = mask
        #
        # # if(train):
        # #     idx = self.srng.choice(size = (k-idx.shape[0]/20,), a = idx, replace = False)
        # #     idx = idx.sort()
        # #idx = theano.tensor.arange(memory.shape[1])
        # splits = memory[idx]






        if (self.mode == "multihopadd"):
            tbr = splits
            for i in range(1, len(self.layers)):
                question = self.layers[i].get_output(train).dimshuffle(0, "x", 1)
                repeated = T.repeat(question, splits.shape[0], axis=1).dimshuffle(1, 0, 2)
                tbr += repeated
            tbr = self.activation(tbr)
            tbr = tbr.dimshuffle(1, 0, 2)

        if (self.mode == "multihobabs"):
            tbr = splits
            for i in range(1, len(self.layers)):
                question = self.layers[i].get_output(train).dimshuffle(0, "x", 1)
                repeated = T.repeat(question, splits.shape[0], axis=1).dimshuffle(1, 0, 2)
                tbr -= repeated

            tbr = abs(tbr)*tbr

            tbr = tbr.dimshuffle(1, 0, 2)
        return tbr

    def get_input(self, train=False):
        res = []
        for i in range(len(self.layers)):
            o = self.layers[i].get_input(train)
            if not type(o) == list:
                o = [o]
            for output in o:
                if output not in res:
                    res.append(output)
        return res


class AttentionRecurrent(Recurrent):
    def __init__(self, output_dim,
                 init='glorot_uniform', inner_init='orthogonal',
                 weights=None, truncate_gradient=-1, return_sequences=False,
                 input_dim=None, input_length=None, **kwargs):
        self.output_dim = output_dim
        self.init = initializations.get(init)
        self.inner_init = initializations.get(inner_init)
        self.inner_activation = activations.get("softmax")
        self.truncate_gradient = truncate_gradient
        self.return_sequences = return_sequences
        self.initial_weights = weights

        self.input_dim = input_dim
        self.input_length = input_length
        self.regularizers = []
        # self.activity_regularizer = AttentionRegulariser2(0.01)


        if self.input_dim:
            kwargs['input_shape'] = (self.input_length, self.input_dim)
        super(AttentionRecurrent, self).__init__(**kwargs)

    def build(self):
        input_dim = self.input_shape[2]
        self.input = T.tensor3()

        self.W = self.inner_init((input_dim, 2))
        self.b = shared_zeros((2))

        self.params = [
            self.W, self.b,
        ]

        if self.initial_weights is not None:
            self.set_weights(self.initial_weights)
            del self.initial_weights


            # self.activity_regularizer.set_layer(self)
            # self.regularizers.append(self.activity_regularizer)

    # def get_attention(self):
    #     X = self.get_input(True)
    #     x_att = self.inner_activation(T.dot(X.dimshuffle(1, 0, 2), self.W) + self.b)[:,0]
    #     return x_att


    def _step(self,
              x_mem, x_att,
              h_tm1
              ):
        z0 = x_att[:, 0].dimshuffle(0, "x")
        z1 = x_att[:, 1].dimshuffle(0, "x")

        h_t = z0 * h_tm1 + z1 * x_mem
        return h_t

    def get_output(self, train=False):
        X = self.get_input(train)

        x_mem = X.dimshuffle((1, 0, 2))

        x_att = self.inner_activation(T.dot(X.dimshuffle(1, 0, 2), self.W) + self.b)

        outputs, updates = theano.scan(
            self._step,
            sequences=[x_mem, x_att],
            outputs_info=[
                T.unbroadcast(alloc_zeros_matrix(X.shape[0], self.output_dim), 1)
            ],
            truncate_gradient=self.truncate_gradient,
            go_backwards=False)
        if self.return_sequences:
            return outputs.dimshuffle((1, 0, 2))
        return outputs[-1]

    def get_config(self):
        config = {"name": self.__class__.__name__,
                  "output_dim": self.output_dim,
                  "init": self.init.__name__,
                  "inner_init": self.inner_init.__name__,
                  "activation": self.activation.__name__,
                  "inner_activation": self.inner_activation.__name__,
                  "truncate_gradient": self.truncate_gradient,
                  "return_sequences": self.return_sequences,
                  "input_dim": self.input_dim,
                  "input_length": self.input_length}
        base_config = super(GRU, self).get_config()
        return dict(list(base_config.items()) + list(config.items()))
