import numpy as np
from rnnLayer import cLM
from datasets import load_data
import time
import sys
import os
import theano


def sgd(learning_rate=0.01, L2_reg=0.0001, n_epochs=1000,
        dataset='../data', batch_size=20, n=300, lambda1=0.001):
    datasets = load_data(dataset)

    train_set_x = datasets[0]
    valid_set_x = datasets[1]
    test_set_x = datasets[2]
    vocab = datasets[3]
   # vocab_dict = datasets[4]

    n_train_batches = len(train_set_x) / batch_size
    n_valid_batches = len(valid_set_x) / batch_size
    n_test_batches = len(test_set_x) / batch_size

    numpy_rng = np.random.RandomState(1234)
    
    V_len = len(vocab)
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

    def train_model(index):
        return clm.train(train_set_x[index * batch_size:
                                     (index + 1) * batch_size],
                         learning_rate, lambda1)

    def validate_model(index):
        return clm.entropy(valid_set_x[index * batch_size:
                                       (index + 1) * batch_size])

    def test_model(index):
        return clm.entropy(valid_set_x[index * batch_size:
                                       (index + 1) * batch_size])
    ###############
    # TRAIN MODEL #
    ###############
    print '... training'

    # early-stopping parameters
    patience = 10000  # look as this many examples regardless
    patience_increase = 2  # wait this much longer when a new best is
                           # found
    improvement_threshold = 0.995  # a relative improvement of this much is
                                   # considered significant
    validation_frequency = min(n_train_batches, patience / 2)
                                  # go through this many
                                  # minibatche before checking the network
                                  # on the validation set; in this case we
                                  # check every epoch

    best_validation_loss = np.inf
    best_iter = 0
    test_score = 0.
    start_time = time.clock()

    epoch = 0
    done_looping = False

    while (epoch < n_epochs) and (not done_looping):
        epoch = epoch + 1
        for minibatch_index in xrange(n_train_batches):

            minibatch_avg_cost = train_model(minibatch_index)
            # iteration number
            iter = (epoch - 1) * n_train_batches + minibatch_index

            if (iter + 1) % validation_frequency == 0:
                # compute zero-one loss on validation set
                validation_losses = [validate_model(i) for i
                                     in xrange(n_valid_batches)]
                this_validation_loss = np.mean(validation_losses)

                print(
                    'epoch %i, minibatch %i/%i, validation error %f %%' %
                    (
                        epoch,
                        minibatch_index + 1,
                        n_train_batches,
                        this_validation_loss * 100.
                    )
                )

                # if we got the best validation score until now
                if this_validation_loss < best_validation_loss:
                    #improve patience if loss improvement is good enough
                    if (
                        this_validation_loss < best_validation_loss *
                        improvement_threshold
                    ):
                        patience = max(patience, iter * patience_increase)

                    best_validation_loss = this_validation_loss
                    best_iter = iter

                    # test it on the test set
                    test_losses = [test_model(i) for i
                                   in xrange(n_test_batches)]
                    test_score = np.mean(test_losses)

                    print(('     epoch %i, minibatch %i/%i, test error of '
                           'best model %f %%') %
                          (epoch, minibatch_index + 1, n_train_batches,
                           test_score * 100.))

            if patience <= iter:
                done_looping = True
                break

    end_time = time.clock()
    print(('Optimization complete. Best validation score of %f %% '
           'obtained at iteration %i, with test performance %f %%') %
          (best_validation_loss * 100., best_iter + 1, test_score * 100.))
    print >> sys.stderr, ('The code for file ' +
                          os.path.split(__file__)[1] +
                          ' ran for %.2fm' % ((end_time - start_time) / 60.))


if __name__ == '__main__':
    sgd()
