import numpy as np

def load_data(dataset_dir):

    if dataset_dir is None:
        raise IOError("Dataset Dir Not Found")

    filenames = [os.path.join(dataset_dir, fname)
                 for fname in ['train', 'valid', 'test']]

    vocab = ['UNK']
    vocab_dict = {'UNK': 0}

    fin = open(filenames[0])
    fval = open(filenames[1])
    ftest = open(filenames[2])

    train = []
    test = []
    valid = []

    if fin is None:
        raise IOError("Filename %s not found" % filename)

    for line in fin:
        words = line.split()
        s = []    

        for word in words:
            if word not in vocab_dict:
                vocab_dict[len(vocab)] = word
                vocab.append(word)

            index = vocab_dict[word]
            s.append(index)
        train.append(np.asarray(s))
    
    val_test_files = [fval, ftest]
    val_test_data = [valid, test]

    for file, data in zip(val_test_files, val_test_data):
        if file is None:
            continue
        for line in fin:
            words = line.split()
            s = []

            for word in words:
                index = vocab_dict[word] 
                            if word in vocab_dict 
                            else 0
                s.append(index)
            data.append(np.asarray(s))

    return (np.asarray(train), 
            np.asarray(valid), 
            np.asarray(test), 
            np.asarray(vocab), 
            vocab_dict)


def build_vocab(filename, L):
    vocab = [None]
    vocab_dict = {0:  None}
    neighbors = []   # neighboring words and n_grams
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
                    neighbors.append([[], []])

                    index += 1

                    prob.append(0.)
                    count.append(0)
                    if i > 0:
                        prefix_id = vocab_dict[" ".join(words[j:j + i])]
                        suffix_id = vocab_dict[words[j + i]]
                        neighbors[prefix_id][1].append(suffix_id)
                        neighbors[suffix_id][0].append(prefix_id)

                id = vocab_dict[phrase]
                count[id] += 1.
                nCount[i] += 1.

                s_l[j] = id

            sentence.append(np.asarray(s_l))
        S.append(np.asarray(sentence))

    for i, phrase in enumerate(vocab[1:]):
        n = len(phrase.split()) - 1
        prob[i] = count[i]/nCount[n]

    return (np.asarray(S), np.asarray(vocab), vocab_dict,
            np.asarray(prob), neighbors)


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
                print "Missing: (%s, ##%s##)" % (word, line)
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
