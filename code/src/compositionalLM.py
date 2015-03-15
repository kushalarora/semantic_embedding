import theano
import numpy as np
import theano.tensor as T
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

        self.compose_params = [self.X, self.W_prob, self.b_prob]

        self.embed_params = []

        hidden_layers = []
        for i in xrange(1, L + 1):
                hidden_layer = HiddenLayer(
                    numpy_rng,
                    self.X,
                    self.P,
                    i,
                    n,
                    self.W_prob,
                    self.b_prob)

                hidden_layers.append(hidden_layer)
                self.embed_params.append(hidden_layer.params)

        self.hidden_layers = theano.shared(
            value=np.asarray(hidden_layers),
            name='hidden_layers')

    def training_fns(self):
        W = T.dmatrix('W')
        learning_rate = T.dscalar(name='lr')

        outputs_info = T.as_tensor_variable(
            np.asarray(0, dtype=theano.config.floatX))

        compose_costs, _ = theano.scan(

            lambda w_l, w_l_1, layer, w_1, total_cost:
                total_cost +
                layer.get_value(borrow=True)
                .get_composition_cost(w_l_1, w_1, w_l),

                sequences=[
                    dict(input=W, taps=[0, -1]),
                    T.arange(W.shape[0]),
                    self.hidden_layers],

                non_sequences=W[0],

                outputs_info=outputs_info)

        embed_costs, _ = theano.scan(

            lambda w_l, w_l_1, layer, w_1, total_cost:
                total_cost +
                layer.get_value(borrow=True)
                .get_embedding_cost(w_l_1, w_1, w_l),

                sequences=[
                    dict(input=W, taps=[0, -1]),
                    T.arange(W.shape[0]),
                    self.hidden_layers],

                non_sequences=W[0],

                outputs_info=outputs_info)

        compose_cost = T.mean(compose_costs)

        embed_cost = T.mean(embed_costs)

        g_compose_params = [T.grad(compose_cost, param)
                            for param in self.compose_params]

        g_embed_params = [T.grad(embed_cost, param)
                          for param in self.embed_params]

        compose_updates = [(param, param - learning_rate * g_param)
                           for param, g_param in zip(self.compose_params,
                                                     g_compose_params)]

        embed_updates = [(param, param - learning_rate * g_param)
                         for param, g_param in zip(self.embed_params,
                                                   g_embed_params)]

        compositional_cost_func = theano.function(
            inputs=[W,
                    theano.Param(learning_rate, default=0.1)],
            outputs=compose_cost,
            updates=compose_updates)

        embedding_cost_func = theano.function(
            inputs=[W,
                    theano.Param(learning_rate, default=0.1)],
            outputs=embed_cost,
            updates=embed_updates)

        return (compositional_cost_func, embedding_cost_func)

    def calculate_prob_embedding_fn(self):
        W = T.ivector('W')

        def oneStep(i, x_m1, p_m1, x_1, p_1):
            layer = self.hidden_layers[i]
            c_fn = layer.get_prob_fn()
            e_fn = layer.get_embedding_fn()
            return (e_fn(x_m1[:-i], x_1),
                    c_fn(x_m1[:-i], x_1, p_m1[:-i], p_1))

        x_1 = self.X[W]
        p_1 = self.P[W]

        [x_vals, p_vals], _ = theano.scan(
            fn=oneStep,
            sequences=[T.arange(W.get_value(borrow=True).shape[0])],
            outputs_info=[dict(input=x_1, taps=[-1]),
                          dict(input=p_1, taps=[-1])],
            non_sequences=[x_1, p_1])

        x_o, p_o = x_vals[-1], p_vals[-1]

        return theano.function(
            [W],
            outputs=(x_o, p_o),
            givens = {
                x_1: self.X[W],
                p_1: self.P[W]
                }
            )


def train(learning_rate=0.13, n=50, L=200, n_epochs=1000,
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
    import pdb
    pdb.set_trace()

    S_train, V, Vindex, P = build_vocab(dataset_train, L)

    S_valid = index_data(dataset_valid, Vindex)

    n_train = len(S_train)

    n_valid = len(S_valid)

    numpy_rng = np.random.RandomState(123)

    V_len = len(V)

    X_initial = np.asarray(
        numpy_rng.uniform(
            low=-4/np.sqrt(6/V_len),
            high=4/np.sqrt(6/V_len),
            size=(n, V_len)
        ),
        dtype=theano.config.floatX)

    X = theano.shared(
        value=X_initial,
        name='X',
        borrow=True)

    cLM = CompositionalLM(
        numpy_rng, X, P, n, L)

    c_fn, e_fn = cLM.training_fns()

    predict_fn = cLM.calculate_prob_embedding_fn()

    ###############
    # TRAIN MODEL #
    ###############
    print '... training the model'

    best_validation_prob = 0

    done_looping = False
    epoch = 0
    while (epoch < n_epochs) and (not done_looping):

        tic = time.time()
        epoch = epoch + 1

        np.random.shuffle(S_train)

        e_cost = 0.0
        for i, sentence in enumerate(S_train):
            e_cost += e_fn(sentence)
            print '[learning embedding] epoch %i >> %2.2f%%' % (
                epoch, (i + 1) * 100. / n_train)
            print 'completed in %.2f (sec) <<\r' % (time.time() - tic),
            sys.stdout.flush()

        tic = time.time()
        c_cost = 0.0
        for i, sentence in enumerate(S_train):
            c_cost += c_fn(sentence)
            print '[learning composition] epoch %i >> %2.2f%%' % (
                epoch, (i + 1) * 100. / n_train)
            print 'completed in %.2f (sec) <<\r' % (time.time() - tic),
            sys.stdout.flush()

        X_valid = []

        tic = time.time()
        valid_prob = 0.0
        for i, sentence in enumerate(S_valid):
            prob, x_embedding = predict_fn(sentence)
            X_valid.append(x_embedding)
            valid_prob *= prob

            print '[learning composition] epoch %i >> %2.2f%%' % (
                epoch, (i + 1) * 100. / n_valid)
            print 'completed in %.2f (sec) <<\r' % (time.time() - tic),
            sys.stdout.flush()

        # if we got the best validation score until now
        if valid_prob > best_validation_prob:

            best_validation_prob = valid_prob
            # test it on the test set

    print(
        (
            'Optimization complete with best validation score of %f %%,'
        )
        % (best_validation_prob)
    )


if __name__ == "__main__":
    train()
