import logging as log
import os
import shutil

from data_formatting.real_world_attributed_networks import RealWorldAttrFormatter
from data_formatting.real_world_non_attributed_networks import RealWorldNonAttrFormatter
from data_formatting.synthetic_networks import SyntheticFormatter

from settings import *

datasets = [
    (SyntheticFormatter, True, {'path': DATA_PATH + 'synthetic/acMark/', 'dataset_name': 'acMark-a=0.1;b=0.1;s=0.1;o=0.1'}),
    (SyntheticFormatter, True, {'path': DATA_PATH + 'synthetic/acMark/', 'dataset_name': 'acMark-a=0.2;b=0.1;s=0.1;o=0.1'}),
    (SyntheticFormatter, True, {'path': DATA_PATH + 'synthetic/acMark/', 'dataset_name': 'acMark-a=0.3;b=0.1;s=0.1;o=0.1'}),
    (SyntheticFormatter, True, {'path': DATA_PATH + 'synthetic/acMark/', 'dataset_name': 'acMark-a=0.4;b=0.1;s=0.1;o=0.1'}),
    (SyntheticFormatter, True, {'path': DATA_PATH + 'synthetic/acMark/', 'dataset_name': 'acMark-a=0.5;b=0.1;s=0.1;o=0.1'}),
    (SyntheticFormatter, True, {'path': DATA_PATH + 'synthetic/acMark/', 'dataset_name': 'acMark-a=0.6;b=0.1;s=0.1;o=0.1'}),
    (SyntheticFormatter, True, {'path': DATA_PATH + 'synthetic/acMark/', 'dataset_name': 'acMark-a=0.7;b=0.1;s=0.1;o=0.1'}),
    (SyntheticFormatter, True, {'path': DATA_PATH + 'synthetic/acMark/', 'dataset_name': 'acMark-a=0.8;b=0.1;s=0.1;o=0.1'}),
    (SyntheticFormatter, True, {'path': DATA_PATH + 'synthetic/acMark/', 'dataset_name': 'acMark-a=0.9;b=0.1;s=0.1;o=0.1'}),
    (SyntheticFormatter, True, {'path': DATA_PATH + 'synthetic/acMark/', 'dataset_name': 'acMark-a=1;b=0.1;s=0.1;o=0.1'}),

    (SyntheticFormatter, False, {'path': DATA_PATH + 'synthetic/LFR/', 'dataset_name': 'LFR-t1=3;t2=1.5;mu=0.1;average_degree=5;'}),
    (SyntheticFormatter, False, {'path': DATA_PATH + 'synthetic/LFR/', 'dataset_name': 'LFR-t1=3;t2=1.5;mu=0.2;average_degree=5;'}),
    (SyntheticFormatter, False, {'path': DATA_PATH + 'synthetic/LFR/', 'dataset_name': 'LFR-t1=3;t2=1.5;mu=0.3;average_degree=5;'}),
    (SyntheticFormatter, False, {'path': DATA_PATH + 'synthetic/LFR/', 'dataset_name': 'LFR-t1=3;t2=1.5;mu=0.4;average_degree=5;'}),
    (SyntheticFormatter, False, {'path': DATA_PATH + 'synthetic/LFR/', 'dataset_name': 'LFR-t1=3;t2=1.5;mu=0.5;average_degree=5;'}),
    (SyntheticFormatter, False, {'path': DATA_PATH + 'synthetic/LFR/', 'dataset_name': 'LFR-t1=3;t2=1.5;mu=0.6;average_degree=5;'}),
    (SyntheticFormatter, False, {'path': DATA_PATH + 'synthetic/LFR/', 'dataset_name': 'LFR-t1=3;t2=1.5;mu=0.7;average_degree=5;'}),
    (SyntheticFormatter, False, {'path': DATA_PATH + 'synthetic/LFR/', 'dataset_name': 'LFR-t1=3;t2=1.5;mu=0.8;average_degree=5;'}),
    (SyntheticFormatter, False, {'path': DATA_PATH + 'synthetic/LFR/', 'dataset_name': 'LFR-t1=3;t2=1.5;mu=0.9;average_degree=5;'}),
    (SyntheticFormatter, False, {'path': DATA_PATH + 'synthetic/LFR/', 'dataset_name': 'LFR-t1=3;t2=1.5;mu=1;average_degree=5;'}),

    (RealWorldAttrFormatter, True, {'path': DATA_PATH + 'real_world_data/', 'dataset_name': 'citeseer'}),
    (RealWorldAttrFormatter, True, {'path': DATA_PATH + 'real_world_data/', 'dataset_name': 'cora'}),
    (RealWorldAttrFormatter, True, {'path': DATA_PATH + 'real_world_data/', 'dataset_name': 'cora_ml'}),
    (RealWorldAttrFormatter, True, {'path': DATA_PATH + 'real_world_data/', 'dataset_name': 'dblp'}),
    (RealWorldAttrFormatter, True, {'path': DATA_PATH + 'real_world_data/', 'dataset_name': 'pubmed'}),

    (RealWorldNonAttrFormatter, False, {'dataset_name': 'Malaria_var_DBLa_HVR_networks_HVR_networks_9'}),
    (RealWorldNonAttrFormatter, False, {'dataset_name': 'Email_network_Uni_R-V_Spain_Email_network_Uni_R-V_Spain'}),
    (RealWorldNonAttrFormatter, False, {'dataset_name': '595b15bd549f067e0263b525'}),
]


def __main__():
    if VEBROSE:
        log.basicConfig(format="%(levelname)s: %(message)s", level=log.INFO)
        log.info("Verbose output.")
    else:
        log.basicConfig(format="%(levelname)s: %(message)s")

    if not os.path.isdir(PRE_COMPUTED_PATH):
        os.makedirs(PRE_COMPUTED_PATH)

    for formatter_cls, attributed, args in datasets:
        data_path = PRE_COMPUTED_PATH + args['dataset_name'] + '/'

        if not os.path.isdir(data_path):
            log.info(f'STARTING PRE-COMPUTING FOR {args["dataset_name"]}')
            data = formatter_cls(args, attributed).load_data()

            log.info('WRITING DATA')

            os.makedirs(data_path)

            data['train_1'].dropna(inplace=True)
            data['test_1'].dropna(inplace=True)
            data['test_2'].dropna(inplace=True)

            data['train_1'].to_csv(data_path + 'train_1.csv')
            data['test_1'].to_csv(data_path + 'test_1.csv')
            data['test_2'].to_csv(data_path + 'test_2.csv')

            log.info('DONE')
        else:
            log.warning(f'Directory {data_path} \nalready exists, skipping...')


__main__()
