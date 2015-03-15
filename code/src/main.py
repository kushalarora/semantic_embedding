import argparse
from datasets import build_vocab

MODE_TRAIN = 0
MODE_TEST = 1


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--mode', type=int,
                        help='Run mode: \n0 => Train, 1 => Test, 2=>BOTH',
                        default=0)
    parser.add_argument('--train', type=str, help='training text file')
    parser.add_argument('--valid', type=str, help='validation text file')
    parser.add_argument('--test', type=str, help='test text file')
    parser.add_argument('--n', type=int,
                        help='dimension of embedding space', default=2)
    parser.add_argument('--L', type=int,
                        help='Maximum length of a sentence(in words)',
                        default=260)

    args = parser.parse_args()
    N = args.n
    MAX_LENGTH = args.L

    if (args.mode & MODE_TRAIN and
       (args.train is None or args.valid is None)):
        raise ValueError('Train or validation file missing')

    if args.mode & MODE_TEST and \
            args.test is None:
        raise ValueError('Test file missing')

    data = build_vocab(args.train, args.L)
