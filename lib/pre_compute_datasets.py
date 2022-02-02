import logging as log
import os
import shutil

from data_formatting.synthetic_networks import SyntheticFormatter

from settings import *

datasets = [
    (SyntheticFormatter,
     True,
     {'path': DATA_PATH + 'synthetic/acMark/', 'dataset_name': 'acMark-a=0.1;b=0.1;s=0.1;o=0.1'})
]


def __main__():
    if VEBROSE:
        log.basicConfig(format="%(levelname)s: %(message)s", level=log.INFO)
        log.info("Verbose output.")
    else:
        log.basicConfig(format="%(levelname)s: %(message)s")

    if os.path.exists(RESULT_PATH) and os.path.isdir(RESULT_PATH):
        shutil.rmtree(RESULT_PATH)

    for formatter_cls, attributed, args in datasets:
        data_path = PRE_COMPUTED_PATH + args['dataset_name'] + '/'
        os.makedirs(data_path)

        log.info(f'STARTING PRE-COMPUTING FOR {args["dataset_name"]}')

        formatter = formatter_cls(args, attributed)

        data = formatter.load_data()

        log.info('WRITING DATA')

        data['train_1'].dropna(inplace=True)
        data['test_1'].dropna(inplace=True)
        data['test_2'].dropna(inplace=True)

        data['train_1'].to_csv(data_path + 'train_1.csv')
        data['test_1'].to_csv(data_path + 'test_1.csv')
        data['test_2'].to_csv(data_path + 'test_2.csv')

        log.info('DONE')


__main__()
