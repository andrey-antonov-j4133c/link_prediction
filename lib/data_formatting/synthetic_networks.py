import numpy as np

from data_formatting.formatter import Formatter
from data_formatting.formatter_opt import OptFormatter


class SyntheticFormatter(Formatter):
    def __init__(self, args: dict, attributed) -> None:
        super().__init__(args, attributed)
        self.A, self.H, self.y = self._read_data({'path': args['path'] + args['dataset_name']})

    def _read_data(self, args):
        n_lines = sum(1 for line in open(args['path'] + '/G.csv')) - 3
        A = np.ndarray((n_lines, n_lines))
        with open(args['path'] + '/G.csv') as f:
            for line in f.readlines():
                if line[0] == '#':
                    continue
                indexes = [int(s) for s in line.split(' ')]
                i, index = indexes[0], indexes[1:]
                for j in range(n_lines):
                    A[i][j] = 1 if j in index else 0

        H = None
        if self.attributed:
            H = []
            with open(args['path'] + '/Features.csv') as f:
                for line in f.readlines():
                    H.append(tuple(float(s) for s in line.split(',')))

        return A, H, None
