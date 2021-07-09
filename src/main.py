from argparse import ArgumentParser
from sys import argv

from experiments import EXPERIMENTS_DICT


if __name__ == '__main__':
    parser = ArgumentParser()
    parser.add_argument('-e', nargs=1, required=True)
    args = parser.parse_args(argv[1:])
    experiment_id = args.e[0]
    EXPERIMENTS_DICT.get(experiment_id, lambda: print('Wrong experiment id.'))()
