import logging as log
import os
import shutil

from SETTINGS import *
from data_generators.real_world_attributed_networks import RealWorldAttrGenerator
from data_generators.synthetic_attributed_networks import SyntheticAttrGenerator
from experiments.experiment import Experiment


def __main__():
    if VEBROSE:
        log.basicConfig(format="%(levelname)s: %(message)s", level=log.INFO)
        log.info("Verbose output.")
    else:
        log.basicConfig(format="%(levelname)s: %(message)s")

    if os.path.exists(RESULT_PATH) and os.path.isdir(RESULT_PATH):
        shutil.rmtree(RESULT_PATH)

    experiments = [
        Experiment(RealWorldAttrGenerator(
            {'path': DATA_PATH + 'real_world_data/', 'dataset_name': 'citeseer'})),
        Experiment(RealWorldAttrGenerator(
            {'path': DATA_PATH + 'real_world_data/', 'dataset_name': 'cora_ml'})),
        Experiment(RealWorldAttrGenerator(
            {'path': DATA_PATH + 'real_world_data/', 'dataset_name': 'cora'})),
        Experiment(RealWorldAttrGenerator(
            {'path': DATA_PATH + 'real_world_data/', 'dataset_name': 'dblp'})),
        Experiment(RealWorldAttrGenerator(
            {'path': DATA_PATH + 'real_world_data/', 'dataset_name': 'pubmed'})),
        Experiment(SyntheticAttrGenerator(
            {'path': DATA_PATH + 'replicated_data/acMark/', 'dataset_name': 'citeseer'})),
        Experiment(SyntheticAttrGenerator(
            {'path': DATA_PATH + 'replicated_data/acMark/', 'dataset_name': 'cora_ml'})),
        Experiment(SyntheticAttrGenerator(
            {'path': DATA_PATH + 'replicated_data/acMark/', 'dataset_name': 'pubmed'})),
        Experiment(SyntheticAttrGenerator(
            {'path': DATA_PATH + 'replicated_data/cabam/', 'dataset_name': 'citeseer'})),
        Experiment(SyntheticAttrGenerator(
            {'path': DATA_PATH + 'replicated_data/cabam/', 'dataset_name': 'cora'})),
        Experiment(SyntheticAttrGenerator(
            {'path': DATA_PATH + 'replicated_data/cabam/', 'dataset_name': 'cora_ml'})),
        Experiment(SyntheticAttrGenerator(
            {'path': DATA_PATH + 'replicated_data/cabam/', 'dataset_name': 'polblogs'})),
        Experiment(SyntheticAttrGenerator(
            {'path': DATA_PATH + 'replicated_data/cabam/', 'dataset_name': 'pubmed'}))
    ]

    for i, e in enumerate(experiments):
        os.makedirs(RESULT_PATH + str(i) + '/')
        e.run(i)
        experiments[i].generator = None


__main__()
