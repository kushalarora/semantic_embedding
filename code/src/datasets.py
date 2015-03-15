import numpy as np


def build_vocab(filename, L):
    vocab = [None]
    vocab_dict = {0:  None}
    prob = [0]
    S = []
    count = [0]
    nCount = [0] * L
    fin = open(filename)
    if fin is None:
        raise IOError("Filename %s not found" % filename)

    index = 1
    for line in fin:

        words = line.split()
        sentence = []

        for i in xrange(len(words)):

            s_l = np.array([0] * len(words))

            for j in xrange(len(words) - i):

                phrase = " ".join(words[j:j + i + 1])

                if phrase not in vocab_dict:

                    vocab.append(phrase)
                    vocab_dict[phrase] = index

                    index += 1

                    prob.append(0.)
                    count.append(0)

                id = vocab_dict[phrase]
                count[id] += 1.
                nCount[i] += 1.

                s_l[j] = id

            sentence.append(np.asarray(s_l))

        S.append(np.asarray(sentence))

    for i, phrase in enumerate(vocab[1:]):
        n = len(phrase.split()) - 1
        prob[i] = count[i]/nCount[n]

    return (np.asarray(S), np.asarray(vocab), vocab_dict, np.asarray(prob))


def index_data(filename, Vindex):
    fin = open(filename)
    if fin is None:
        raise IOError("Filename %s not found" % filename)

    if Vindex is None:
        raise BaseException("Vocab index not found")

    S = []
    for line in fin:

        words = line.split()
        sentence = []

        missing_vocab = False
        for word in words:
            if word not in Vindex:
                missing_vocab = True
                break

        if missing_vocab:
            continue

        for word in words:

            try:
                id = Vindex[word]
            except KeyError:
                continue

            if id is None:
                raise RuntimeError("Word: %s missing in vocabulary" % word)

            sentence.append(id)

        S.append(np.asarray(sentence))

    return np.asarray(S)
