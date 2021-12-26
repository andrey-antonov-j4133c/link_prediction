import os
import shutil

from SETTINGS import * 
import logging as log

from Experiment.experiment import Experiment
from Generator.RealWorldAttributedNetworkGenerator import RealWorldAttributedNetorkGeberator

def __main__():
    if VEBROSE:
        log.basicConfig(format="%(levelname)s: %(message)s", level=log.DEBUG)
        log.info("Verbose output.")
    else:
        log.basicConfig(format="%(levelname)s: %(message)s")

    if os.path.exists(RESULT_PATH) and os.path.isdir(RESULT_PATH):
        shutil.rmtree(RESULT_PATH)

    experiments = [
        Experiment(RealWorldAttributedNetorkGeberator(), {'path': DATA_PATH, 'dataset_name': 'citeseer'}),
        Experiment(RealWorldAttributedNetorkGeberator(), {'path': DATA_PATH, 'dataset_name': 'cora_ml'}),
        Experiment(RealWorldAttributedNetorkGeberator(), {'path': DATA_PATH, 'dataset_name': 'cora'}),
        Experiment(RealWorldAttributedNetorkGeberator(), {'path': DATA_PATH, 'dataset_name': 'dblp'}),
        Experiment(RealWorldAttributedNetorkGeberator(), {'path': DATA_PATH, 'dataset_name': 'pubmed'})
    ]

    for i, e in enumerate(experiments):
        os.makedirs(RESULT_PATH + str(i) + '/')
        e.run(i)

__main__()