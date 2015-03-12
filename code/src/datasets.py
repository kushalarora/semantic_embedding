def build_vocab(filename):
    vocab = []
    vocab_dict = {}
    prob = []
    X = []
    count = []
    nCount = []
    fin = open(filename)
    if fin is None:
        raise IOError("Filename %s not found" % filename)

    index = 0
    for line in fin:

        print "... processing line %s" % line
        words = line.split()
        sentence = []

        for i in xrange(len(words)):

            s_l = []
            nCount.append(0)

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

                prob[id] = count[id]/nCount[i]

                s_l.append(id)

            sentence.append(s_l)

        X.append(sentence)

    return (X, vocab, vocab_dict, prob)
