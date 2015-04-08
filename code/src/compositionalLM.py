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
                 M,
                 n=50,
                 L=200,
                 W_prob=None,
                 b_prob=None,
                 W_L=None,
                 b_L=None):

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

        if W_L is None:
            initial_W_L = np.asarray(
                numpy_rng.uniform(
                    low=-4 * np.sqrt(6./(3 * n + L)),
                    high=4 * np.sqrt(6./(3 * n + L)),
                    size=(L, n, 2*n)),
                dtype=theano.config.floatX)

            W_L = theano.shared(
                value=initial_W_L,
                name='W_L',
                borrow=True)

        if b_L is None:
            b_L = theano.shared(
                np.asarray(
                    numpy_rng.uniform(
                        low=-4 * np.sqrt(6/(n + L)),
                        high=4 * np.sqrt(6/(n + L)),
                        size=(L, n)),
                    dtype=theano.config.floatX),
                name='b_L',
                borrow=True)

        self.W_prob = W_prob

        self.b_prob = b_prob

        self.W_L = W_L

        self.b_L = b_L

        self.X = X

        self.P = P

        self.M = M

        self.n = n

        self.L = L

    def training_fns(self):
        return HiddenLayer.training_fns(self.W_L, self.b_L,
                                        self.W_prob, self.b_prob,
                                        self. X, self.P)

    def prob_embedding_fn(self):
        prob_fn = HiddenLayer.probability_function(self.W_prob,
                                                   self.b_prob)
        embed_fn = HiddenLayer.embedding_function(self.W_L,
                                                  self.b_L)

        return (prob_fn, embed_fn)


def train(learning_rate=0.0, n=50, L=200, n_epochs=50,
          dataset_train='../data/train', dataset_valid='../data/valid',
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

    S_train, V, Vindex, P, neighbors = build_vocab(dataset_train, L)
    S_valid = index_data(dataset_valid, Vindex)

    n_train = len(S_train)

    n_valid = len(S_valid)

    numpy_rng = np.random.RandomState(123)

    V_len = len(V)

    X_initial = np.asarray(
        numpy_rng.uniform(
            low=-4 * np.sqrt(6./V_len),
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
    tic = time.time()

    M_initial = []
    for neighbor in neighbors:
        m = []
        for neigh in neighbor:
            if len(neigh) == 0:
                m.append(np.zeros((n)))
            else:
                m.append(np.mean(X_initial[neigh], axis=0))
        M_initial.append(m)

    M = theano.shared(
        value=np.asarray(M_initial),
        name='M',
        borrow=True)

    print ".. Building Model"

    cLM = CompositionalLM(
        numpy_rng, X, P, M, n, L)

    composition_train_fn, entropy_train_fn = cLM.training_fns()

    prob_fn, embed_fn = cLM.prob_embedding_fn()

    print ".. Building model completed in %2.2f secs" % (time.time() - tic)
    ###############
    # TRAIN MODEL #
    ###############
    print '... training the model'

    best_validation_pp = np.inf

    done_looping = False
    epoch = 0
    while (epoch < n_epochs) and (not done_looping):
        print "Epoch Number: %d of %d" % (epoch, n_epochs)
        tic = time.time()
        epoch = epoch + 1

        # np.random.shuffle(S_train)
        te_cost = 0.0
        tc_cost = 0.0
        import pdb
        for i, sentence in enumerate(S_train):
            try:
                s_len = len(sentence)
                if s_len < 2:
                    continue

                c_costs, e_costs = (entropy_train_fn(sentence, s_len,
                                                     0.2, learning_rate),
                                    composition_train_fn(sentence, s_len,
                                                         learning_rate))
                if np.isnan(np.min(c_costs)) or np.isnan(np.min(e_costs)):
                    pdb.set_trace()
                print('[learning embedding]' +
                      '%2.2d/%2.2d in %.2f (sec) costs>>(%2.2f, %2.2f)\r' % (
                          (i + 1), n_train, time.time() - tic,
                          c_costs[-1], e_costs[-1])),

                te_cost += e_costs[-1]
                tc_cost += c_costs[-1]
            except:
                pdb.set_trace()

        print('[learning embedding] ' +
              '# %2.2d completed in %2.2f (sec) T cost >> (%2.2f, %2.2f)<<' % (
                  i, time.time() - tic, tc_cost, te_cost))

        # tic = time.time()
        # tc_cost = 0.0
        # for i, sentence in enumerate(S_train):
        #     c_cost = 0.0
        #     for j in xrange(len(sentence) - 1):
        #         cost, _ = c_fns[j](sentence[j],
        #                            sentence[0],
        #                            sentence[j+1],
        #                            learning_rate)
        #         c_cost += np.sqrt(cost)
        #     tc_cost += c_cost

        #     print(
        #         '[learning composition] epoch %i >> %2.2f%' % (
        #             epoch, (i + 1) * 100. / n_train) +
        #         'completed in %.2f (sec) cost >> %2.2f <<\r' % (
        #             time.time() - tic, c_cost)),
        #     sys.stdout.flush()

        # print(
        #     '[learning composition] epoch %i >> %2.2f%' % (
        #         epoch, (i + 1) * 100. / n_train) +
        #     'completed in %.2f (sec) T cost >> %2.2f <<' % (
        #         time.time() - tic, tc_cost))

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

            for j in xrange(1, len(sentence)):
                x = embed_fn(j, x, x0)
                p = prob_fn(j, x, x0, p, p0)
            t /= pow(p[0], 1./len(sentence))

            print('[validation]' +
                  '%2.2d/%2.2d completed in %.2f (sec) cost >> %2.2f <<\r' % (
                      (i + 1), n_train, time.time() - tic, t)),
            sys.stdout.flush()

        valid_pp = pow(t, 1./n_valid)
        print(
            '[validation]' +
            'epoch %d completed in %.2f (sec) T cost >> %2.2f <<' % (
                epoch, time.time() - tic, valid_pp))

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
    train(n=10, L=50)
