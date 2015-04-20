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
    def _probabilities(l, W_prob, b_prob,
                       X_l_1, X_1, P_l_1, P_1):
        """ Returns probability for level l.
            The probability is computed as

            (p_i)^l = H(G(X_i^{l-1}, X_{i+l}^1; W_prob, b_prob),
                        P_i^{l-1}, P_1)

            P(w_i^l/w_i^{l-1}w_{i+1}^1) = sigmoid(w_1^T * W_prob * w_2 +
                                                  b_prob)

            (p_i)^l = P(w_i^l/w_i^{l-1}w_{i+1}^1) * p_i^{l-1} * p_{i+l}^1
        """
        P_y_x, updates = theano.scan(

            lambda x_i, x_i_l:
                T.nnet.sigmoid(
                    T.dot(T.dot(x_i, W_prob), x_i_l) + b_prob),

            sequences=[X_l_1, X_1])

        P_le, updates = theano.scan(

            lambda p_y_x, p_i, p_i_l: p_y_x * p_i * p_i_l,

            sequences=[P_y_x, P_l_1, P_1])

        return pad_prob(P_le, P_1.shape[0])

    @staticmethod
    def _embeddings(l, W_l, b_l, X_l_1, X_1):
        """ Returns the embedding X_i^l from X_i^{l-1} and
            X_{i+l}^1

            X_i^l = W_l[X_i^{l-1} X_{i+l}^1]^T + b_l

        """
        embeddings, updates = theano.scan(
            lambda x_l_1, x_1:
                T.tanh(T.dot(W_l,
                             T.concatenate([x_l_1, x_1])) + b_l),
            sequences=[X_l_1, X_1])

        return pad_embedding(embeddings, X_1.shape[0])

    @staticmethod
    def _embedding_cost(l, X_l, M_l):
        costs, _ = theano.scan(lambda x_l, m_l, prior_cost:
                               prior_cost + T.sum(T.sqr(x_l - m_l)),
                               sequences=[X_l, M_l])
        return costs[-1]

    @staticmethod
    def _composition_cost(l, W_l, b_l, X):
        """ Squared error minimization error
            training objective
        """
        X_1 = X[HiddenLayer.W[0]][-l:]
        X_l_1 = X[HiddenLayer.W[l-1]][:l]

        X_l = X[HiddenLayer.W[l]][:l]

        X_le = HiddenLayer._embeddings(l, W_l, b_l, X_l_1, X_1)
        return T.sqrt(T.sum((X_l - X_le[:l])**2))

    @staticmethod
    def _cross_entropy_cost(l, W_prob, b_prob, X, P):
        """ Cross entropy minimization objective function for level l.
            Doing L2 regularization
            Summing P_i^l over i for normalization.
        """
        X_1 = X[HiddenLayer.W[0]][-l:]
        X_l_1 = X[HiddenLayer.W[l-1]][:l]

        P_1 = P[HiddenLayer.W[0]][-l:]
        P_l_1 = P[HiddenLayer.W[l-1]][:l]

        P_l = P[HiddenLayer.W[l]][:l]

        P_le = HiddenLayer._probabilities(l, W_prob, b_prob,
                                          X_l_1, X_1, P_l_1, P_1)

        L = T.nnet.categorical_crossentropy(P_le[:l],
                                            P_l[:l]) + \
            P_le[:-l].sum() + \
            HiddenLayer.lambda1 * (W_prob ** 2).sum()

        return T.mean(L)

    @staticmethod
    def probability_function(l, W_prob, b_prob):
        """ Returns function to get probability for n-gram
            w_i^l from w_i^{l-1} and w_{i+l}^1
        """
        return theano.function(
            inputs=[HiddenLayer.X_l_1, HiddenLayer.X_1,
                    HiddenLayer.P_l_1, HiddenLayer.P_1],
            outputs=HiddenLayer._probabilities(l, W_prob,
                                               b_prob,
                                               HiddenLayer.X_l_1[:l],
                                               HiddenLayer.X_1[-l:],
                                               HiddenLayer.P_l_1[:l],
                                               HiddenLayer.P_1[-l:]),
            )

    @staticmethod
    def embedding_function(l, W_l, b_l):
        """ Returns function to get embedding for n-gram X_i^l
            from X_i^{l-1} and X_{i+l}^1
        """
        return theano.function(
            inputs=[HiddenLayer.X_l_1, HiddenLayer.X_1],
            outputs=HiddenLayer._embeddings(l, W_l, b_l,
                                            HiddenLayer.X_l_1[:l],
                                            HiddenLayer.X_1[:-l]))

    @staticmethod
    def cost_fns(W_l, b_l, W_prob, b_prob, X, P):

        l = T.iscalar('l')      # layer number

        entropy_fn = theano.function(
            [HiddenLayer.W, l, HiddenLayer.lambda1],
            outputs=HiddenLayer._cross_entropy_cost(l, W_prob, b_prob,
                                                    X, P))
        comp_fn = theano.function(
            inputs=[HiddenLayer.W, l],
            outputs=HiddenLayer._composition_cost(l, W_l, b_l, X))

        return (comp_fn, entropy_fn)

    @staticmethod
    def training_fns(W_L, b_L, W_prob, b_prob, X, P):
        """ Returns composition function
        """
        l = T.iscalar('l')      # layer number

        lr = HiddenLayer.lr     # learning rate

        Ls = T.arange(1, l)     # range for ls (1, .. l-1)

        print "..Computing expression for composition cost"

        composition_cost, _ = theano.scan(
            lambda i, W_i, b_i, cost, X:
                cost + HiddenLayer._composition_cost(i, W_i, b_i, X),
            sequences=[Ls, W_L, b_L],
            outputs_info=T.as_tensor_variable(
                np.asarray(0, dtype=theano.config.floatX)),
            non_sequences=X)

        print "..Computing expression for entropy cost"

        entropy, _ = theano.scan(
            lambda i, cost, W_prob, b_prob, X, P:
                cost + HiddenLayer._cross_entropy_cost(i, W_prob, b_prob,
                                                       X, P),
            sequences=[Ls],
            outputs_info=T.as_tensor_variable(
                np.asarray(0, dtype=theano.config.floatX)),
            non_sequences=[W_prob, b_prob, X, P])

        print ".. Computing updates for composition"

        composition_updates = [(param,
                                param - lr * T.grad(composition_cost[-1],
                                                    param))
                               for param in [W_L, b_L]]

        print ".. Computing updates for entropy"

        entropy_updates = [(param, param - lr * T.grad(entropy[-1], param))
                           for param in [W_prob, b_prob, X]]

        print ".. Building composition function"
        composition_function = theano.function(
            inputs=[HiddenLayer.W, l, lr],
            outputs=composition_cost,
            updates=composition_updates)

        print ".. Building entropy function"

        entropy_function = theano.function(
            inputs=[HiddenLayer.W, l, lr, HiddenLayer.lambda1],
            outputs=entropy,
            updates=entropy_updates)

        return (composition_function, entropy_function)


def pad_embedding(seq, length):
    pad_size = length - seq.shape[0]

    padding = T.zeros((pad_size, seq.shape[1]))

    return T.concatenate([seq, padding], axis=0)


def pad_prob(seq, length):
    pad_size = length - seq.shape[0]

    padding = T.zeros((pad_size,))

    return T.concatenate([seq, padding], axis=0)

if __name__ == '__main__':
    n = 2   # dimension of embedding space
    N = 6   # Number of examples
    L = 10   # Number of layers
    X = theano.shared(
        np.asarray(
            np.random.uniform(
                high=1,
                low=-1,
                size=(N, n))),
        borrow=True)

    P = theano.shared(
        np.asarray(
            np.random.uniform(
                low=0.01,
                high=0.1,
                size=(N,))),
        borrow=True)

    initial_W_L = np.asarray(
        np.random.uniform(
            low=-4 * np.sqrt(6./(3 * n + L)),
            high=4 * np.sqrt(6./(3 * n + L)),
            size=(L, n, 2 * n)),
        dtype=theano.config.floatX)

    W_L = theano.shared(
        value=initial_W_L,
        name='W_L',
        borrow=True)

    b_L = theano.shared(
        np.asarray(
            np.random.uniform(
                low=-4 * np.sqrt(6/(n + L)),
                high=4 * np.sqrt(6/(n + L)),
                size=(L, n)),
            dtype=theano.config.floatX),
        name='b_L',
        borrow=True)

    initial_W_prob = np.asarray(
        np.random.uniform(
            low=-4 * np.sqrt(3/n),
            high=4 * np.sqrt(3/n),
            size=(n, n)),
        dtype=theano.config.floatX)

    W_prob = theano.shared(
        value=initial_W_prob,
        name='W_prob',
        borrow=True)

    b_prob = theano.shared(
        value=0.0,
        name='b_prob')

    fe = HiddenLayer.embedding_function(1, W_L[0], b_L[0])
    print fe([[1, 0], [1, 0], [1, 0]],
             [[1, 0], [1, 0], [1, 0]])

    fp = HiddenLayer.probability_function(1, W_prob, b_prob)
    print fp(np.matrix([[1, 0], [1, 0], [1, 0]]),
             np.matrix([[1, 0], [1, 0], [1, 0]]),
             [0.5, 0.5, 0.5],
             [0.1, 0.1, 0.1])

    comp_cost, entropy_cost = HiddenLayer.cost_fns(W_L[0], b_L[0],
                                                   W_prob, b_prob, X, P)
    fet, fpt = HiddenLayer.training_fns(W_L, b_L,
                                        W_prob, b_prob, X, P)
    comp_cost = fpt([[0, 1, 2],
                     [1, 2, 3],
                     [1, 2, 3],
                     [1, 2, 3],
                     [3, 4, 5]],
                    4, 0.0, 0.1)

    embed_cost = fet([[0, 1, 2],
                      [1, 2, 3],
                      [1, 2, 3],
                      [1, 2, 3],
                      [3, 4, 5]],
                     4, 0.1)

    print "Comp Cost: %s\t Embed Cost: %s\r" % (comp_cost, embed_cost)
