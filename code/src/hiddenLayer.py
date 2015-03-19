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

    W = T.lmatrix('W')

    lambda1 = T.dscalar('lambda1')

    lr = T.dscalar('lr')

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
        X_1 = X[HiddenLayer.W[0]]
        X_l_1 = X[HiddenLayer.W[l-1]]
        P_1 = P[HiddenLayer.W[0]]
        P_l_1 = P[HiddenLayer.W[l-1]]
        P_l = P[HiddenLayer.W[l]]
        P_le = HiddenLayer._get_comp_probability(l, W_prob, b_prob,
                                                 X_l_1, X_1, P_l_1, P_1)

        L = T.nnet.categorical_crossentropy(P_le[:-l],
                                            P_l[:-l]) + \
            P_le[:-l].sum() + \
            HiddenLayer.lambda1 * (W_prob ** 2).sum()

        return T.mean(L)

    @staticmethod
    def _get_embedding_cost(l, W_l, b_l, X):
        X_1 = X[HiddenLayer.W[0]]
        X_l_1 = X[HiddenLayer.W[l-1]]
        X_l = X[HiddenLayer.W[l]]
        X_le = HiddenLayer._get_embeddings(l, W_l, b_l, X_l_1, X_1)
        return T.sqrt(T.sum((X_l[:-l] - X_le[:-l])**2))

    def __init__(self,
                 numpy_rng,
                 X,
                 P,
                 composite_embed_cost=None,
                 composite_comp_cost=None,
                 composite_embed_params=[],
                 composite_comp_params=[],
                 composite_g_comp_params=[],
                 composite_g_embed_params=[],
                 l=1,
                 n=50,
                 W_prob=None,
                 b_prob=None):

        self.l = l
        self.X = X
        self.P = P

        self.composite_comp_params = T.copy(composite_comp_params)
        self.composite_embed_params = T.copy(composite_embed_params)
        self.composite_g_comp_params = T.copy(composite_g_comp_params)
        self.composite_g_embed_params = T.copy(composite_g_embed_params)

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
            name='b_%d' % l,
            borrow=True)

        W_l_initial = np.asarray(
            numpy_rng.uniform(
                low=-4*np.sqrt(3/n),
                high=4*np.sqrt(3/n),
                size=(n, 2*n)),
            dtype=theano.config.floatX)

        self.W_l = theano.shared(value=W_l_initial, name='W_%d' % l,
                                 borrow=True)

        self.comp_params = [self.W_prob, self.b_prob, self.X]

        self.embed_params = [self.W_l, self.b_l]

        self.embed_cost = HiddenLayer._get_embedding_cost(l, self.W_l,
                                                          self.b_l, self.X)

        self.comp_cost = HiddenLayer._get_composition_cost(l, self.W_prob,
                                                           self.b_prob,
                                                           self.X, self.P)

        self.composite_embed_cost = self.embed_cost

        if composite_embed_cost is not None:
            self.composite_embed_cost += composite_embed_cost

        self.composite_comp_cost = self.comp_cost

        if composite_comp_cost is not None:
            self.composite_comp_cost += composite_comp_cost

        self.composite_embed_params.extend(self.embed_params)

        if len(self.composite_comp_params) == 0:
            self.composite_comp_cost.extend(self.comp_cost)

        g_comp_params = [T.grad(self.comp_cost, param)
                         for param in self.comp_params]

        g_embed_params = [T.grad(self.embed_cost, param)
                          for param in self.embed_cost]

        self.composite_g_embed_params.extend(g_embed_params)

        self.composite_g_comp_params += g_comp_params

    def composite_train_fns(self):

        comp_updates = [(param, param - HiddenLayer.lr * gparam)
                        for param, gparam in zip(self.composite_comp_params,
                                                 self.composite_g_comp_params)]

        embed_updates = [(param, param - HiddenLayer.lr * gparam)
                         for param, gparam in zip(self.composite_embed_params,
                                                  self.composite_g_embed_params)]

        embed_train_fn = theano.function([HiddenLayer.W, HiddenLayer.lr],
                                         outputs=self.composite_embed_cost,
                                         updates=embed_updates)

        comp_train_fn = theano.function([HiddenLayer.W, HiddenLayer.lambda1,
                                         HiddenLayer.lr],
                                        outputs=self.composite_comp_cost,
                                        updates=comp_updates)
        return (comp_train_fn, embed_train_fn)

    def train_fns(self):

        g_comp_params = [T.grad(self.comp_cost, param)
                         for param in self.comp_params]

        g_embed_params = [T.grad(self.embed_cost, param)
                          for param in self.embed_params]

        comp_updates = [(param, param - HiddenLayer.lr * gparam)
                        for param, gparam in zip(self.comp_params,
                                                 g_comp_params)]

        embed_updates = [(param, param - HiddenLayer.lr * gparam)
                         for param, gparam in zip(self.embed_params,
                                                  g_embed_params)]

        embed_train_fn = theano.function([HiddenLayer.W, HiddenLayer.lr],
                                         outputs=self.embed_cost,
                                         updates=embed_updates)

        comp_train_fn = theano.function([HiddenLayer.W, HiddenLayer.lambda1,
                                         HiddenLayer.lr],
                                        outputs=self.comp_cost,
                                        updates=comp_updates)
        return (comp_train_fn, embed_train_fn)


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
    import pdb;pdb.set_trace()

    import pdb;pdb.set_trace()
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

    fpt, fet = hl.train_fns()

    comp_cost = fpt([[0, 1, 2],
                     [1, 2, 3],
                     [3, 4, 5]],
                    0.0, 0.1)

    embed_cost = fet([[0, 1, 2],
                      [1, 2, 3],
                      [3, 4, 5]],
                     0.1)

    print "Comp Cost: %s\t Embed Cost: %s\r" % (comp_cost, embed_cost)
