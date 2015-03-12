import theano
import theano.tensor as T
import numpy as np


class HiddenLayer:
    """ This is the lth layer in our compositional network. It builds
        phrase of length l by combining sub-phrases of length l-1 and a word.
        This is done in two steps. In the Embed Step, we learn the embedding of
        phrases from l-1 layer and 0 layer(words) that minimizes cross entropy
        between the original and emtimated probability of phrase of length l.

        In second stage, the Compose stage, we learn the compositional operator
        by minimizing reconstruction error between the embedding learning
        in previous step and through composition."""
    def __init__(self,
                 numpy_rng,
                 X,
                 P,
                 l=1,
                 n=50,
                 W_prob=None,
                 b_prob=None):

        self.w_l = T.ivector(name='w_l')
        self.w_l_1 = T.ivector(name='w_l_1')
        self.w_1 = T.ivector(name='w_1')

        if self.w_l_1.ndim != self.w_1.ndim:
            raise TypeError('Both inputs should be of same dimension')

        if self.w_l.ndim != self.w_1.ndim:
            raise TypeError('Inputs and composed phrases should be  \
                of same dimension')

        self.X = X
        self.P = P

        self.W_prob = W_prob

        self.b_prob = b_prob

        self.b_l = theano.shared(
            value=np.zeros((n,), dtype=theano.config.floatX),
            name='b_l',
            borrow=True)

        W_l_initial = np.asarray(
            numpy_rng.uniform(
                low=-4*np.sqrt(3/n),
                high=4*np.sqrt(3/n)),
            size=(n, n),
            dtype=theano.config.floatX)

        self.W_l = theano.shared(value=W_l_initial, name='W_l', borrow=True)

        self.params = [self.W_l, self.b_l]

        def get_compositional_probability(self):
            X_l_1 = X[self.w_l_1]
            X_1 = X[self.w_1]

            P_l_1 = P[self.w_l_1]
            P_1 = P[self.w_1]

            P_y_x, updates = theano.scan(

                lambda x_i, x_i_l:
                    T.nnet.sigmoid(
                        T.dot(T.dot(x_i, self.W_prob), x_i_l) + self.b_prob),

                sequences=[X_l_1, X_1])

            P_le, updates = theano.scan(

                lambda p_y_x, p_i, p_i_l: p_y_x * p_i * p_i_l,

                sequences=[P_y_x, P_l_1, P_1])

            return P_le

        def get_embeddings(self):
            X_l_1 = self.X[self.w_l_1]
            X_1 = self.X[self.w_1]

            return T.tanh(
                theano.scan(
                    lambda x_l_1, x_1:
                        T.dot(self.W_l, T.concatenate(x_l_1, x_1, axis=1)),
                    sequences=[X_l_1, X_1]))

        def get_composition_cost_updates(self, learning_rate):
            P_lo = P[self.w_l]

            P_le = self.get_compositional_probability()

            L = T.nnet.categorical_crossentropy(P_le, P_lo)

            cost = T.mean(L)

            g_W_prob = T.grad(cost, self.W_prob)
            g_X = T.grad(cost, self.X)

            updates = [(self.W_prob, self.W_prob - learning_rate * g_W_prob),
                       (self.X, self.X - learning_rate * g_X)]
            return (cost, updates)

        def get_embedding_cost_updates(self, learning_rate):
            X_lo = X[self.w_l]

            X_le = self.get_embeddings()

            cost = np.sum((X_lo, X_le)**2)

            g_W_l = T.grad(cost, self.W_l)

            updates = [(self.W_l, self.W_l - learning_rate * g_W_l)]
            return (cost, updates)


class CompositionalLM:
    def __init__(self,
                 numpy_rng,
                 X,
                 P,
                 n=50,
                 L=200,
                 W_prob=None,
                 b_prob=None):

        if W_prob is None:
            initial_W_prob = np.asarray(
                numpy_rng.uniform(
                    low=-4 * np.sqrt(3/n),
                    high=4 * np.sqrt(3/n)),
                size=(n, n),
                dtype=theano.config.floatX),

            W_prob = theano.shared(
                value=initial_W_prob,
                name='W_prob',
                borrow=True)

        if b_prob is None:
            b_prob = theano.shared(
                value=0.0,
                name='b_prob',
                borrow=True)

        self.W_prob = W_prob

        self.b_prob = b_prob

        self.hidden_layers = []

        self.params = [self.W_prob, self.b_prob]

        for i in xrange(1, L + 1):
                hidden_layer = HiddenLayer(
                    numpy_rng,
                    X,
                    P,
                    i,
                    n,
                    W_prob,
                    b_prob)

                self.hidden_layers.append(hidden_layer)

    def training_fns(self, learning_rate):
        compositional_funcs = []
        for layer in self.hidden_layers:
            cost, updates = layer.get_compositional_cost_updates(learning_rate)

            func = theano.function(
                inputs=[self.w_l, self.w_l_1, self.w_1,
                        theano.Param(learning_rate, default=0.1)],
                outputs=cost,
                updates=updates)

            compositional_funcs.append(func)

        embedding_funcs = []
        for layer in self.hidden_layers:
            cost, updates = layer.get_embedding_cost_updates(learning_rate)

            func = theano.function(
                inputs=[self.w_l, self.w_l_1, self.w_1,
                        theano.Param(learning_rate, default=0.1)],
                outputs=cost,
                updates=updates)

            embedding_funcs.append(func)
        return (compositional_funcs, embedding_funcs)


def train():
    pass
