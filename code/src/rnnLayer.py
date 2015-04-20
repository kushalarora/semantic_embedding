import theano
import numpy as np
import theano.tensor as T

sentence = T.ivector()
beta = T.dscalar()
W = T.imatrix()
lr = T.dscalar()
lambda1 = T.dscalar()


class cLM(object):
    def __init__(self, n, X, numpy_rng):
        initial_W_prob = np.asarray(
            numpy_rng.uniform(
                low=-4 * np.sqrt(6./n),
                high=4 * np.sqrt(6./n),
                size=(n,)),
            dtype=theano.config.floatX)

        self.W_prob = theano.shared(
            value=initial_W_prob,
            name='W_prob',
            borrow=True)

        self.b_prob = theano.shared(
            value=0.0,
            name='b_prob')

        initial_W_L = np.array(
            numpy_rng.uniform(
                low=-4 * np.sqrt(2./n),
                high=4 * np.sqrt(2./n),
                size=(n, 2*n)),
            dtype=theano.config.floatX)

        self.W_L = theano.shared(
            value=initial_W_L,
            name='W_L',
            borrow=True)

        self.b_L = theano.shared(
            value=np.zeros(
                (n,),
                dtype=theano.config.floatX),
            name='b_L')

        self.X = X

        self.params = [self.W_prob, self.b_prob, self.W_L,
                       self.b_L, self.X]

        self.L2 = T.sum([T.sum(param ** 2)
                         for param in self.params])

        self.training_fn = None
        self.normalize_fn = None

    def compose(self, x_0, x_l_1):
        return T.dot(self.W_L, T.concatenate([x_0, x_l_1])) + self.b_L

    def energy(self, x):
        return T.dot(self.W_prob.T, x) + self.b_prob

    def composition(self, l, X_0, X_l_1):
        N = X_0.shape[0]
        X_l, _ = theano.scan(self.compose,
                             sequences=[X_l_1[:N-l], X_0[-N+l:]])
        return pad_right(X_l, N)

    def energies(self, X_l):
        E_l, _ = theano.scan(self.energy,
                             sequences=[X_l])
        return E_l

    def energy_sentence(self, sentence):
        X_0 = self.X[sentence]

        X_S, _ = theano.scan(
            lambda i, X_l_1, X_0:
                self.composition(i, X_0, X_l_1),
            sequences=[T.arange(sentence.shape[0])],
            outputs_info=X_0,
            non_sequences=X_0)

        energy_sentence, _ = theano.scan(
            lambda X_l, energy:
                energy + T.sum(self.energies(X_l)),
            sequences=[X_S],
            outputs_info=T.as_tensor_variable(
                np.asarray(0, dtype=theano.config.floatX)))

        return energy_sentence[-1]

    def prob_w_given_h(self, h, w):
        E_s, _ = theano.scan(
            lambda x, h:
                self.energy(
                    self.compose(h, x)),
                sequences=[self.X],
                non_sequences=h)
        e_w = self.energy(
            self.compose(h, w))
        return T.exp(-1 * beta * (e_w - T.sum(E_s)))

    def training_fn_builder(self):
        total_energy, _ = theano.scan(
            lambda w, energy:
                energy + self.energy_sentence(w),
            sequences=[W],
            outputs_info=T.as_tensor_variable(
                np.asarray(0, dtype=theano.config.floatX)))

        loss_func = total_energy[-1] + lambda1 * self.L2
        updates = [(param,
                    param - lr * T.grad(loss_func, param))
                   for param in self.params]
        return theano.function(
            [W, lr, lambda1],
            total_energy[-1],
            updates=updates)

    def normalize_func_builder(self):
        return theano.function(
            [],
            updates={self.X:
                     self.X/T.sqrt(T.sum(self.X**2, axis=1))
                     .dimshuffle(0, 'x')})

    def train(self,
              sentences,
              learning_rate=0.1,
              lmbda=0.01):
        if self.training_fn is None and self.normalize_fn is None:
            self.training_fn = self.training_fn_builder()
            self.normalize_fn = self.normalize_func_builder()

        energy = self.training_fn(sentences, learning_rate, lmbda)
        self.normalize_fn()
        return energy

    def entropy(self, W):
        word_count = W.shape[0] * W.shape[1]

        def sentence_entropy(sentence):
            entropy, _ = theano.scan(
                lambda i, sentence, entropy:
                    entropy - T.log(self.prob_w_given_h(
                        sentence[:i], sentence[i]))/beta,

                sequences=[T.arange(sentence.shape[0])],
                outputs_info=T.as_tensor_variable(
                    np.asarray(0, dtype=theano.config.floatX)))
            return entropy[-1]

        entropies, _ = theano.scan(
            lambda w, entropy: entropy + sentence_entropy(w),
            sequences=[W],
            outputs_info=T.as_tensor_variable(
                np.asarray(0, dtype=theano.config.floatX)))

        return entropies[-1]/word_count


def pad_right(seq, length):
    pad_size = length - seq.shape[0]

    padding = T.zeros((pad_size, seq.shape[1]))

    return T.concatenate([seq, padding], axis=0)

if __name__ == "__main__":
    numpy_rng = np.random.RandomState(1234)
    V_len = 10
    n = 5
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

    clm = cLM(n, X, numpy_rng)

    p = clm.compose(X[0], X[1])

    e = clm.energy(X[0])

    P = clm.composition(0, X[[1, 2, 3]], X[[4, 5]])

    E = clm.energies(X[[1, 2, 3]])

    ES = clm.energy_sentence(np.array([1, 2, 3]))

    pwh = clm.prob_w_given_h(X[1], X[3])

    clm.normalize = clm.normalize_func_builder()

    clm.normalize()
    import pdb
    pdb.set_trace()

    clm.train([[1, 2, 3]])
