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

        self.X_l_1 = T.dmatrix(name='X_l_1')
        self.X_1 = T.dmatrix(name='X_1')

        self.P_l_1 = T.dvector(name='P_l_1')
        self.P_1 = T.dvector(name='P_1')

        if self.X_1.ndim != self.X_l_1.ndim or \
                self.P_1.ndim != self.P_l_1.ndim:
                    raise TypeError('Layer %i: Both inputs should be of \
                                    same dimension' % l)

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

        self.comp_params = [self.W_prob, self.b_prob, self.X]

    def _get_compositional_probability(self, X_l_1, X_1, P_l_1, P_1):
        P_y_x, updates = theano.scan(

            lambda x_i, x_i_l:
                T.nnet.sigmoid(
                    T.dot(T.dot(x_i, self.W_prob), x_i_l) +
                    self.b_prob),

            sequences=[dict(input=X_l_1, taps=[-self.l]),
                       dict(input=X_1, taps=[self.l])])

        P_le, updates = theano.scan(

            lambda p_y_x, p_i, p_i_l: p_y_x * p_i * p_i_l,

            sequences=[P_y_x, dict(input=P_l_1, taps=[-self.l]),
                       dict(input=P_1, taps=[self.l])])

        return pad_prob(P_le, P_1.shape[0])

    def get_prob_fn(self):
        return theano.function(
            inputs=[self.X_l_1, self.X_1, self.P_l_1, self.P_1],
            outputs=self._get_compositional_probability(self.X_l_1,
                                                        self.X_1,
                                                        self.P_l_1,
                                                        self.P_1))

    def get_embedding_fn(self):
        return theano.function(
            inputs=[self.X_l_1, self.X_1],
            outputs=self._get_embeddings(self.X_l_1,
                                         self.X_1))

    def _get_embeddings(self, X_l_1, X_1):
        embeddings, updates = theano.scan(
            lambda x_l_1, x_1:
                T.tanh(T.dot(self.W_l,
                             T.concatenate([x_l_1, x_1])) + self.b_l),
            sequences=[dict(input=X_l_1, taps=[-self.l]),
                       dict(input=X_1, taps=[self.l])])

        return pad_embedding(embeddings, X_1.shape[0])

    def get_composition_training_fn(self):
        """
            w_l_1: TensorVariable :: phrases of length l-1
            w_1: TensorVariable :: words of length 1
            w_l: TensorVariable :: phrases of length l

        """

        w_l_1 = T.ivector('w_l_1')
        w_1 = T.ivector('w_1')
        w_l = T.ivector('w_l')

        X_l_1 = self.X[w_l_1]
        X_1 = self.X[w_1]
        P_l_1 = self.P[w_l_1]
        P_1 = self.P[w_1]

        P_l = self.P[w_l]

        lr = T.dscalar('lr')

        P_le = self._get_compositional_probability(X_l_1, X_1, P_l_1, P_1)

        L = T.nnet.categorical_crossentropy(P_le, P_l)

        cost = T.mean(L)

        gparams = [T.grad(cost, param)
                   for param in self.comp_params]

        updates = [(param, param - gparam * lr)
                   for param, gparam in zip(self.comp_params, gparams)]

        return theano.function(
            [w_l_1, w_1, w_l, lr],
            outputs=cost,
            updates=updates)

    def get_embedding_train_fn(self):

        w_l_1 = T.ivector('w_l_1')
        w_1 = T.ivector('w_1')
        w_l = T.ivector('w_l')

        X_l_1 = self.X[w_l_1]
        X_1 = self.X[w_1]
        X_l = self.X[w_l]
        X_le = self._get_embeddings(X_l_1, X_1)

        lr = T.dscalar('lr')

        cost = T.sum(T.sqr(X_l - X_le))

        gparams = [T.grad(cost, param)
                   for param in self.embed_params]

        updates = [(param, param - gparam * lr)
                   for param, gparam in zip(self.embed_params, gparams)]

        return theano.function(
            [w_l_1, w_1, w_l, lr],
            outputs=cost,
            updates=updates
            )


def pad_embedding(seq, length):
    pad_size = length - seq.shape[0]

    padding = T.zeros((pad_size, seq.shape[1]))

    return T.concatenate([seq, padding], axis=0)


def pad_prob(seq, length):
    pad_size = length - seq.shape[0]

    padding = T.zeros((pad_size,))

    return T.concatenate([seq, padding], axis=0)

if __name__ == '__main__':

    hl = HiddenLayer(
        np.random.RandomState(3),
        None,
        None,
        n=2)

    print "W_prob: %s\tb_prob: %s\tW_l: %s\tb_l: %s " % (
        hl.W_prob.get_value(),
        hl.b_prob.get_value(),
        hl.W_l.get_value(),
        hl.b_l.get_value())

    import pdb
    pdb.set_trace()

    fe = hl.get_embedding_fn()
    print fe([[1, 0], [1, 0], [1, 0], [1, 0]],
             [[1, 0], [1, 0], [1, 0], [1, 0]])

    fp = hl.get_prob_fn()
    print fp(np.matrix([[1, 0], [1, 0], [1, 0]]),
             np.matrix([[1, 0], [1, 0], [1, 0]]),
             [0.5, 0.5, 0.5],
             [0.1, 0.1, 0.1])
