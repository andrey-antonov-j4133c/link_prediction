import logging as log
import numpy as np
from scipy.sparse import csr_matrix

from generator import Generator


class RealWorldAttrGenerator(Generator):
    def __init__(self, args: dict) -> None:
        super().__init__(args)
        self.A, self.H, self.y = self._read_data({'path': args['path'] + args['dataset_name']})

    def _read_data(self, args: dict):
        log.info('Reading data ...')
        path = args['path']
        if not path.endswith('.npz'):
            path += '.npz'
        with np.load(path, allow_pickle=True) as loader:
            loader = dict(loader)
            A = csr_matrix((loader['adj_data'], loader['adj_indices'],
                            loader['adj_indptr']), shape=loader['adj_shape'])

            H = csr_matrix((loader['attr_data'], loader['attr_indices'],
                            loader['attr_indptr']), shape=loader['attr_shape'])

            y = loader.get('labels')

            log.info('Success!')

            return A.toarray(), [tuple(i) for i in H.toarray()], y
