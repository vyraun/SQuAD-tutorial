from __future__ import print_function

import sys

from load import Loader


def main():
    pass

if __name__ == '__main__':
    if(len(sys.argv) <= 2):
        raise ValueError('Give location of dataset train and dev json file.')
    train_data = Loader(sys.argv[1]).get_data()
    dev_data = Loader(sys.argv[1]).get_data()
    main()
