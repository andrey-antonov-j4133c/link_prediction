import logging as log
import os
import shutil

import ray

from data_formatting.real_world_attributed_networks import RealWorldAttrFormatter
from data_formatting.real_world_non_attributed_networks import RealWorldNonAttrFormatter
from data_formatting.synthetic_networks import SyntheticFormatter

from settings import *

data = []

for a in ('0.1', '0.2', '0.3', '0.4', '0.5', '0.6', '0.7', '0.8', '0.9', '1'):
    for run in (1, 2, 3, 4, 5):
        data.append(
            (SyntheticFormatter, True,
             {'path': DATA_PATH + 'synthetic/acMark/', 'dataset_name': f'acMark-a={a};b=0.1;s=0.1;o=0.1_run{run}'})
        )

data.append(
    (RealWorldAttrFormatter, True, {'path': DATA_PATH + 'real_world_data/', 'dataset_name': f'citeseer.npz'})
)


def __main__():
    ray.init()

    if VEBROSE:
        log.basicConfig(format="%(levelname)s: %(message)s", level=log.INFO)
        log.info("Verbose output.")
    else:
        log.basicConfig(format="%(levelname)s: %(message)s")

    if not os.path.isdir(PRE_COMPUTED_PATH):
        os.makedirs(PRE_COMPUTED_PATH)

    for formatter_cls, attributed, args in data:
        data_path = PRE_COMPUTED_PATH + args['dataset_name'] + '/'

        if not os.path.isdir(data_path):
            try:
                log.info(f'STARTING PRE-COMPUTING FOR {args["dataset_name"]}')
                os.makedirs(data_path)

                train1, test1, test2, attributes = formatter_cls(args, attributed).load_data(pre_compute=True)

                log.info('WRITING DATA')

                train1.to_csv(data_path + 'train_1.csv')
                test1.to_csv(data_path + 'test_1.csv')
                test2.to_csv(data_path + 'test_2.csv')

                if attributed:
                    attributes.to_csv(data_path + 'attributes.csv')

                log.info('DONE')
            except BaseException as e:
                log.error(e)
                log.error('Failed to compute, try again later')

                if os.path.exists(data_path) and os.path.isdir(data_path):
                    shutil.rmtree(data_path)
        else:
            log.warning(f'Directory\n{data_path} \nalready exists, skipping...')


__main__()
