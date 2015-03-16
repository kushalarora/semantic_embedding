import theano
import numpy as np
# import theano.tensor as T
import time
import sys
from hiddenLayer import HiddenLayer
from datasets import build_vocab, index_data


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

        self.X = X

        self.P = P

        self.hidden_layers = []
        for i in xrange(1, L + 1):
            hidden_layer = HiddenLayer(
                numpy_rng,
                self.X,
                self.P,
                i,
                n,
                self.W_prob,
                self.b_prob)

            self.hidden_layers.append(hidden_layer)

    def training_fns(self):
        comp_train_fns = []
        embed_train_fns = []

        for i, layer in enumerate(self.hidden_layers):
            print ".. Building train model for layer: %i" % i
            comp_train_fns.append(layer.get_composition_training_fn())
            embed_train_fns.append(layer.get_embedding_train_fn())

        return (comp_train_fns, embed_train_fns)

    def prob_embedding_fn(self):
        prob_fns = []
        embed_fns = []

        for i, layer in enumerate(self.hidden_layers):
            print ".. Building valid model for layer: %i" % i
            prob_fns.append(layer.get_prob_fn())
            embed_fns.append(layer.get_embedding_fn())

        return (prob_fns, embed_fns)


def train(learning_rate=0.13, n=50, L=200, n_epochs=50,
          dataset_train='../data/train1', dataset_valid='../data/valid1',
          batch_size=600):
    """
    Demonstrate stochastic gradient descent optimization of a log-linear
    model

    This is demonstrated on MNIST.

    :type learning_rate: float
    :param learning_rate: learning rate used (factor for the stochastic
                          gradient)

    :type n_epochs: int
    :param n_epochs: maximal number of epochs to run the optimizer

    :type dataset: string
    :param dataset: the path of the MNIST dataset file from
     http://www.iro.umontreal.ca/~lisa/deep/data/mnist/mnist.pkl.gz
     """

    S_train, V, Vindex, P = build_vocab(dataset_train, L)

    S_valid = index_data(dataset_valid, Vindex)

    n_train = len(S_train)

    n_valid = len(S_valid)


    numpy_rng = np.random.RandomState(123)

    V_len = len(V)

    X_initial = np.asarray(
        numpy_rng.uniform(
            low=-4 *np.sqrt(6./V_len),
            high=4 * np.sqrt(6./V_len),
            size=(V_len, n)
        ),
        dtype=theano.config.floatX)

    X = theano.shared(
        value=X_initial,
        name='X',
        borrow=True)

    P = theano.shared(
        value=P,
        name='P',
        borrow=True)

    cLM = CompositionalLM(
        numpy_rng, X, P, n, L)

    e_fns, c_fns = cLM.training_fns()

    prob_fns, embed_fns = cLM.prob_embedding_fn()

    ###############
    # TRAIN MODEL #
    ###############
    print '... training the model'

    best_validation_pp = np.inf

    done_looping = False
    epoch = 0
    while (epoch < n_epochs) and (not done_looping):

        tic = time.time()
        epoch = epoch + 1

        np.random.shuffle(S_train)
        te_cost = 0.0
        for i, sentence in enumerate(S_train):
            e_cost = 0.0
            for j in xrange(len(sentence) - 1):
                cost, _ = e_fns[j](sentence[j],
                                   sentence[0],
                                   sentence[j+1],
                                   learning_rate,
                                   0.2)
                e_cost += cost
            te_cost += e_cost

            print '[learning embedding] epoch %i >> %2.2f%% completed in %.2f (sec) cost >> %2.2f <<\r' % (
                epoch, (i + 1) * 100. / n_train, time.time() - tic, e_cost),
            sys.stdout.flush()

        print '[learning embedding] epoch %i >> %2.2f%% completed in %.2f (sec) T cost >> %2.2f <<\r' % (
            epoch, (i + 1) * 100. / n_train, time.time() - tic, te_cost)
        sys.stdout.flush()

        tic = time.time()
        tc_cost = 0.0
        for i, sentence in enumerate(S_train):
            c_cost = 0.0
            for j in xrange(len(sentence) - 1):
                cost, _ = c_fns[j](sentence[j],
                                   sentence[0],
                                   sentence[j+1],
                                   learning_rate)
                c_cost += np.sqrt(cost)
            tc_cost += c_cost

            print '[learning composition] epoch %i >> %2.2f%% completed in %.2f (sec) cost >> %2.2f <<\r' % (
                epoch, (i + 1) * 100. / n_train, time.time() - tic, c_cost),
            sys.stdout.flush()

        print '[learning composition] epoch %i >> %2.2f%% completed in %.2f (sec) T cost >> %2.2f <<' % (
            epoch, (i + 1) * 100. / n_train, time.time() - tic, tc_cost)

        tic = time.time()
        t = 1.0
        X_vals = X.get_value(borrow=True)
        P_vals = P.get_value(borrow=True)
        print "..validating model"
        for i, sentence in enumerate(S_valid):
            x = X_vals[sentence]
            x0 = X_vals[sentence]
            p = P_vals[sentence]
            p0 = P_vals[sentence]

            for j in xrange(len(sentence) - 1):
                x = embed_fns[j](x, x0)
                p = prob_fns[j](x, x0, p, p0)
            t /= pow(p[0], 1./len(sentence))

            print '[validation] epoch %i >> %2.2f%% completed in %.2f (sec) cost >> %2.2f <<\r' % (
                epoch, (i + 1) * 100. / n_train, time.time() - tic, t),
            sys.stdout.flush()

        valid_pp = pow(t, 1./n_valid)
        print '[validation] epoch %i >> %2.2f%% completed in %.2f (sec) T cost >> %2.2f <<' % (
            epoch, (i + 1) * 100. / n_train, time.time() - tic, valid_pp)

        # if we got the best validation score until now
        if valid_pp < best_validation_pp:
            print "current best pp at epoch: %d: %2.2f%%" % (epoch, valid_pp)
            best_validation_pp = valid_pp
            # test it on the test set
    print(
        (
            'Optimization complete with best validation score of %f %%,'
        )
        % (best_validation_pp)
    )


if __name__ == "__main__":
    train(n=10, L=10)
