import logging as log
import os
import shutil
import ray

from models.nn_model import NNModel
from models.gb_model import GBModel

from settings import *

from data_formatting.real_world_attributed_networks import RealWorldAttrFormatter
from data_formatting.synthetic_networks import SyntheticFormatter
from data_formatting.real_world_non_attributed_networks import RealWorldNonAttrFormatter

from experiments.experiment import Experiment


def __main__():
    ray.init()

    if VEBROSE:
        log.basicConfig(format="%(levelname)s: %(message)s", level=log.INFO)
        log.info("Verbose output.")
    else:
        log.basicConfig(format="%(levelname)s: %(message)s")

    if os.path.exists(RESULT_PATH) and os.path.isdir(RESULT_PATH):
        shutil.rmtree(RESULT_PATH)

    #experiments = [
    #    Experiment(RealWorldAttrFormatter(
    #        {'path': DATA_PATH + 'real_world_data/', 'dataset_name': 'citeseer'})),
    #    Experiment(RealWorldAttrFormatter(
    #        {'path': DATA_PATH + 'real_world_data/', 'dataset_name': 'cora_ml'})),
    #    Experiment(RealWorldAttrFormatter(
    #        {'path': DATA_PATH + 'real_world_data/', 'dataset_name': 'cora'})),
    #    Experiment(RealWorldAttrFormatter(
    #        {'path': DATA_PATH + 'real_world_data/', 'dataset_name': 'dblp'})),
    #    Experiment(RealWorldAttrFormatter(
    #        {'path': DATA_PATH + 'real_world_data/', 'dataset_name': 'pubmed'})),
    #    Experiment(SyntheticAttrFormatter(
    #        {'path': DATA_PATH + 'replicated_data/acMark/', 'dataset_name': 'citeseer'})),
    #    Experiment(SyntheticAttrFormatter(
    #        {'path': DATA_PATH + 'replicated_data/acMark/', 'dataset_name': 'cora_ml'})),
    #    Experiment(SyntheticAttrFormatter(
    #        {'path': DATA_PATH + 'replicated_data/acMark/', 'dataset_name': 'pubmed'})),
    #    Experiment(SyntheticAttrFormatter(
    #        {'path': DATA_PATH + 'replicated_data/cabam/', 'dataset_name': 'citeseer'})),
    #    Experiment(SyntheticAttrFormatter(
    #        {'path': DATA_PATH + 'replicated_data/cabam/', 'dataset_name': 'cora'})),
    #    Experiment(SyntheticAttrFormatter(
    #        {'path': DATA_PATH + 'replicated_data/cabam/', 'dataset_name': 'cora_ml'})),
    #    Experiment(SyntheticAttrFormatter(
    #        {'path': DATA_PATH + 'replicated_data/cabam/', 'dataset_name': 'polblogs'})),
    #    Experiment(SyntheticAttrFormatter(
    #        {'path': DATA_PATH + 'replicated_data/cabam/', 'dataset_name': 'pubmed'}))
    #]

    data = [
        #(RealWorldNonAttrFormatter, Experiment, GBModel,
        # False, {'dataset_name': 'Malaria_var_DBLa_HVR_networks_HVR_networks_9'}),
        #(RealWorldNonAttrFormatter, Experiment, GBModel,
        # False, {'dataset_name': 'Email_network_Uni_R-V_Spain_Email_network_Uni_R-V_Spain'}),
        #(RealWorldNonAttrFormatter, Experiment, GBModel,
        # False, {'dataset_name': '595b15bd549f067e0263b525'}),
        #(RealWorldNonAttrFormatter, Experiment, NNModel,
        # False, {'dataset_name': 'Malaria_var_DBLa_HVR_networks_HVR_networks_9'}),
        #(RealWorldNonAttrFormatter, Experiment, NNModel,
        # False, {'dataset_name': 'Email_network_Uni_R-V_Spain_Email_network_Uni_R-V_Spain'}),
        #(RealWorldNonAttrFormatter, Experiment, NNModel,
        # False, {'dataset_name': '595b15bd549f067e0263b525'}),

        #(RealWorldAttrFormatter, Experiment, NNModel,
        # True, {'path': DATA_PATH + 'real_world_data/', 'dataset_name': 'citeseer'}),
        #(RealWorldAttrFormatter, Experiment, NNModel,
        # False, {'path': DATA_PATH + 'real_world_data/', 'dataset_name': 'citeseer'}),
        #(RealWorldAttrFormatter, Experiment, GBModel,
        # False, {'path': DATA_PATH + 'real_world_data/', 'dataset_name': 'citeseer'}),

        #(RealWorldNonAttrFormatter, Experiment, NNModel,
        # False, {'dataset_name': 'Malaria_var_DBLa_HVR_networks_HVR_networks_9'}),
        #(RealWorldNonAttrFormatter, Experiment, GBModel,
        # False, {'dataset_name': 'Malaria_var_DBLa_HVR_networks_HVR_networks_9'}),

        (SyntheticFormatter, Experiment, NNModel,
         True, {'path': DATA_PATH + 'synthetic/acMark/', 'dataset_name': 'acMark-a=0.1;b=0.1;s=0.1;o=0.1'}),
        #(SyntheticFormatter, Experiment, NNModel,
        # False, {'path': DATA_PATH + 'synthetic/acMark/', 'dataset_name': 'acMark-a=0.1;b=0.1;s=0.1;o=0.1'}),
        (SyntheticFormatter, Experiment, GBModel,
         False, {'path': DATA_PATH + 'synthetic/acMark/', 'dataset_name': 'acMark-a=0.1;b=0.1;s=0.1;o=0.1'}),

        #(SyntheticFormatter, Experiment, NNModel,
        # False, {'path': DATA_PATH + 'synthetic/LFR/', 'dataset_name': 'LFR-t1=3;t2=1.5;mu=0.1;average_degree=5;'}),
        #(SyntheticFormatter, Experiment, GBModel,
        # False, {'path': DATA_PATH + 'synthetic/LFR/', 'dataset_name': 'LFR-t1=3;t2=1.5;mu=0.1;average_degree=5;'}),

    ]

    for i, (formatter, experiment, model, attributed, args) in enumerate(data):
        experiment_path = f"exp #{i+1}, {args['dataset_name']}, attributed is {attributed}, model: {model.MODEL_TYPE}"
        os.makedirs(RESULT_PATH + experiment_path + '/')

        log.info('STARTING EXPERIMENT')
        log.info(experiment_path)

        e = experiment(formatter(args, attributed), model)
        e.run(attributed, experiment_path)


__main__()
