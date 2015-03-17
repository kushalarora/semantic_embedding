import theano.tensor as T
import theano
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

    X_l_1 = T.dmatrix(name='X_l_1')
    X_1 = T.dmatrix(name='X_1')
    X_l = T.dmatrix(name='X_l')

    P_l_1 = T.dvector(name='P_l_1')
    P_1 = T.dvector(name='P_1')
    P_l = T.dvector(name='P_l')

    w_l_1 = T.ivector(name='w_l_1')
    w_l = T.ivector(name='w_l')
    w_1 = T.ivector(name='w_1')

    W_l = T.dmatrix('W_l')
    b_l = T.dvector('b_l')

    W_prob = T.dmatrix('W_prob')
    b_prob = T.dscalar('b_prob')

    lambda1 = T.dscalar('lambda1')

    def __init__(self,
                 numpy_rng,
                 X,
                 P,
                 l=1,
                 n=50,
                 W_prob=None,
                 b_prob=None):

        self.l = l
        self.X = X
        self.P = P

        if W_prob is None:
            initial_W_prob = np.asarray(
                numpy_rng.uniform(
                    low=-4 * np.sqrt(3/n),
                    high=4 * np.sqrt(3/n),
                    size=(n, n)),
                dtype=theano.config.floatX)

            W_prob = theano.shared(
                value=initial_W_prob,
                name='W_prob',
                borrow=True)

        if b_prob is None:
            b_prob = theano.shared(
                value=0.0,
                name='b_prob')

        self.W_prob = W_prob

        self.b_prob = b_prob

        self.b_l = theano.shared(
            value=np.zeros((n,), dtype=theano.config.floatX),
            name='b_l',
            borrow=True)

        W_l_initial = np.asarray(
            numpy_rng.uniform(
                low=-4*np.sqrt(3/n),
                high=4*np.sqrt(3/n),
                size=(n, 2*n)),
            dtype=theano.config.floatX)

        self.W_l = theano.shared(value=W_l_initial, name='W_l', borrow=True)

        self.embed_params = [self.W_l, self.b_l]

        self.comp_params = [self.W_prob, self.b_prob]

        self.embed_cost = HiddenLayer._get_embedding_cost(l, self.W_l,
                                                          self.b_l, self.X)

        self.comp_cost = HiddenLayer._get_composition_cost(l, self.W_prob,
                                                           self.b_prob,
                                                           self.X, self.P)

    def train_fn(self):
        W = T.dmatrix('W')

        fpt = theano.function(
            inputs=[HiddenLayer.w_l_1, HiddenLayer.w_1, HiddenLayer.w_l,
                    HiddenLayer.lambda1],
            outputs=hl.comp_cost,
            givens={
                HiddenLayer.X_l_1: X[HiddenLayer.w_l_1],
                HiddenLayer.X_1: X[HiddenLayer.w_1],
                HiddenLayer.P_l_1: P[HiddenLayer.w_l_1],
                HiddenLayer.P_1: P[HiddenLayer.w_1],
                HiddenLayer.P_l: P[HiddenLayer.w_l],
                }
            )

        fet = theano.function(
            inputs=[HiddenLayer.w_l_1, HiddenLayer.w_1, HiddenLayer.w_l],
            outputs=hl.embed_cost,
            givens={
                HiddenLayer.X_l_1: X[HiddenLayer.w_l_1],
                HiddenLayer.X_1: X[HiddenLayer.w_1],
                HiddenLayer.X_l: X[HiddenLayer.w_l]
                }
            )

        comp_costs, _ = theano.scan(
            lambda w_l_1, w_l, w_1: fpt(w_l_1, w_1, w_l),
            sequences=[dict(inputs=W, taps=[-1, 0])],
            non_sequences=[W[0]])

        embed_costs, _ = theano.scan(
            lambda w_l_1, w_l, w_1: fet(w_l_1, w_1, w_l),
            sequences=[dict(inputs=W, taps=[-1, 0])],
            non_sequences=[W[0]])

    @staticmethod
    def _get_comp_probability(l, W_prob, b_prob,
                              X_l_1, X_1, P_l_1, P_1):
        P_y_x, updates = theano.scan(

            lambda x_i, x_i_l:
                T.nnet.sigmoid(
                    T.dot(T.dot(x_i, W_prob), x_i_l) + b_prob),

            sequences=[dict(input=X_l_1, taps=[-l]),
                       dict(input=X_1, taps=[l])])

        P_le, updates = theano.scan(

            lambda p_y_x, p_i, p_i_l: p_y_x * p_i * p_i_l,

            sequences=[P_y_x,
                       dict(input=P_l_1, taps=[-l]),
                       dict(input=P_1, taps=[l])])

        return pad_prob(P_le, P_1.shape[0])

    @staticmethod
    def _get_embeddings(l, W_l, b_l, X_l_1, X_1):
        embeddings, updates = theano.scan(
            lambda x_l_1, x_1:
                T.tanh(T.dot(W_l,
                             T.concatenate([x_l_1, x_1])) + b_l),
            sequences=[dict(input=X_l_1, taps=[-l]),
                       dict(input=X_1, taps=[l])])

        return pad_embedding(embeddings, X_1.shape[0])

    @staticmethod
    def get_prob_fn(l, W_prob, b_prob):
        return theano.function(
            inputs=[HiddenLayer.X_l_1, HiddenLayer.X_1,
                    HiddenLayer.P_l_1, HiddenLayer.P_1],
            outputs=HiddenLayer._get_comp_probability(l, W_prob,
                                                      b_prob,
                                                      HiddenLayer.X_l_1,
                                                      HiddenLayer.X_1,
                                                      HiddenLayer.P_l_1,
                                                      HiddenLayer.P_1),
            )

    @staticmethod
    def get_embedding_fn(l, W_l, b_l):
        return theano.function(
            inputs=[HiddenLayer.X_l_1, HiddenLayer.X_1],
            outputs=HiddenLayer._get_embeddings(l, W_l, b_l,
                                                HiddenLayer.X_l_1,
                                                HiddenLayer.X_1))

    @staticmethod
    def _get_composition_cost(l, W_prob, b_prob, X, P):
        X_1 = X[HiddenLayer.w_1]
        X_l_1 = X[HiddenLayer.w_l_1]
        P_1 = P[HiddenLayer.w_1]
        P_l_1 = P[HiddenLayer.w_l_1]
        P_l = P[HiddenLayer.w_l]
        P_le = HiddenLayer._get_comp_probability(l, W_prob, b_prob,
                                                 X_l_1, X_1, P_l_1, P_1)

        L = T.nnet.categorical_crossentropy(P_le[:-l],
                                            P_l[:-l]) + \
            P_le[:-l].sum() + \
            HiddenLayer.lambda1 * (W_prob ** 2).sum()

        return T.mean(L)

    @staticmethod
    def _get_embedding_cost(l, W_l, b_l, X):
        X_1 = X[HiddenLayer.w_1]
        X_l_1 = X[HiddenLayer.w_l_1]
        X_l = X[HiddenLayer.w_l]
        X_le = HiddenLayer._get_embeddings(l, W_l, b_l, X_l_1, X_1)
        return T.sqrt(T.sum((X_l[:-l] - X_le[:-l])**2))


def pad_embedding(seq, length):
    pad_size = length - seq.shape[0]

    padding = T.zeros((pad_size, seq.shape[1]))

    return T.concatenate([seq, padding], axis=0)


def pad_prob(seq, length):
    pad_size = length - seq.shape[0]

    padding = T.zeros((pad_size,))

    return T.concatenate([seq, padding], axis=0)

if __name__ == '__main__':
    X = theano.shared(
        np.asarray(
            np.random.uniform(
                high=1,
                low=-1,
                size=(6, 2))),
        borrow=True)

    P = theano.shared(
        np.asarray(
            np.random.uniform(
                low=0.01,
                high=0.1,
                size=(6,))),
        borrow=True)

    hl = HiddenLayer(
        np.random.RandomState(3),
        X,
        P,
        n=2)

    print "W_prob: %s\tb_prob: %s\tW_l: %s\tb_l: %s " % (
        hl.W_prob.get_value(),
        hl.b_prob.get_value(),
        hl.W_l.get_value(),
        hl.b_l.get_value())

    fe = HiddenLayer.get_embedding_fn(1, hl.W_l, hl.b_l)
    print fe([[1, 0], [1, 0], [1, 0]],
             [[1, 0], [1, 0], [1, 0]])

    fp = HiddenLayer.get_prob_fn(1, hl.W_prob, hl.b_prob)
    print fp(np.matrix([[1, 0], [1, 0], [1, 0]]),
             np.matrix([[1, 0], [1, 0], [1, 0]]),
             [0.5, 0.5, 0.5],
             [0.1, 0.1, 0.1])

    fet = theano.function(
        inputs=[HiddenLayer.w_l_1, HiddenLayer.w_1, HiddenLayer.w_l],
        outputs=hl.embed_cost)

    fpt = theano.function(
        inputs=[HiddenLayer.w_l_1, HiddenLayer.w_1, HiddenLayer.w_l,
                HiddenLayer.lambda1],
        outputs=hl.comp_cost)

    cost = fpt([0, 1, 2],
               [1, 2, 3],
               [3, 4, 5], 0.0)

    print "Cost: %s\t\r" % (cost)
