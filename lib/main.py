import os
import shutil

from SETTINGS import * 
import logging as log

from Experiment.experiment import Experiment
from Generator.RealWorldAttributedNetworkGenerator import RealWorldAttributedNetorkGeberator
from Generator.ReplicatedAttributedNetworkGenerator import ReplicatedAttributedNetworkGenerator

def __main__():
    if VEBROSE:
        log.basicConfig(format="%(levelname)s: %(message)s", level=log.INFO)
        log.info("Verbose output.")
    else:
        log.basicConfig(format="%(levelname)s: %(message)s")

    if os.path.exists(RESULT_PATH) and os.path.isdir(RESULT_PATH):
        shutil.rmtree(RESULT_PATH)

    experiments = [
        #Experiment(RealWorldAttributedNetorkGeberator({'path': DATA_PATH + 'real_world_data/', 'dataset_name': 'citeseer'})),
        #Experiment(RealWorldAttributedNetorkGeberator({'path': DATA_PATH + 'real_world_data/', 'dataset_name': 'cora_ml'})),
        #Experiment(RealWorldAttributedNetorkGeberator({'path': DATA_PATH + 'real_world_data/', 'dataset_name': 'cora'})),
        #Experiment(RealWorldAttributedNetorkGeberator({'path': DATA_PATH + 'real_world_data/', 'dataset_name': 'dblp'})),
        #Experiment(RealWorldAttributedNetorkGeberator({'path': DATA_PATH + 'real_world_data/', 'dataset_name': 'pubmed'})),
        Experiment(ReplicatedAttributedNetworkGenerator({'path': DATA_PATH+'replicated_data/acMark/', 'dataset_name': 'citeseer'})),
        #Experiment(ReplicatedAttributedNetworkGenerator({'path': DATA_PATH+'replicated_data/acMark/', 'dataset_name': 'cora_ml'})),
        #Experiment(ReplicatedAttributedNetworkGenerator({'path': DATA_PATH+'replicated_data/acMark/', 'dataset_name': 'pubmed'})),
        #Experiment(ReplicatedAttributedNetworkGenerator({'path': DATA_PATH+'replicated_data/cabam/', 'dataset_name': 'citeseer'})),
        #Experiment(ReplicatedAttributedNetworkGenerator({'path': DATA_PATH+'replicated_data/cabam/', 'dataset_name': 'cora'})),
        #Experiment(ReplicatedAttributedNetworkGenerator({'path': DATA_PATH+'replicated_data/cabam/', 'dataset_name': 'cora_ml'})),
        #Experiment(ReplicatedAttributedNetworkGenerator({'path': DATA_PATH+'replicated_data/cabam/', 'dataset_name': 'polblogs'})),
        #Experiment(ReplicatedAttributedNetworkGenerator({'path': DATA_PATH+'replicated_data/cabam/', 'dataset_name': 'pubmed'}))
    ]

    for i, e in enumerate(experiments):
        os.makedirs(RESULT_PATH + str(i) + '/')
        e.run(i)
        experiments[i].generator = None

__main__()