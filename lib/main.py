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
from experiments.feature_selection_experiment import FeatureSelectionExperiment


def __main__():
    ray.init()

    if VEBROSE:
        log.basicConfig(format="%(levelname)s: %(message)s", level=log.INFO)
        log.info("Verbose output.")
    else:
        log.basicConfig(format="%(levelname)s: %(message)s")

    #if os.path.exists(RESULT_PATH) and os.path.isdir(RESULT_PATH):
    #    shutil.rmtree(RESULT_PATH)

    data = [
        # LEGACY EXPERIMENTS #
        # 1. LFR mu 0.1-1.0
        (SyntheticFormatter, Experiment, GBModel,
         False, {'path': DATA_PATH + 'synthetic/LFR/', 'dataset_name': 'LFR-t1=3;t2=1.5;mu=0.1;average_degree=5;'}),
        (SyntheticFormatter, Experiment, GBModel,
         False, {'path': DATA_PATH + 'synthetic/LFR/', 'dataset_name': 'LFR-t1=3;t2=1.5;mu=0.2;average_degree=5;'}),
        (SyntheticFormatter, Experiment, GBModel,
         False, {'path': DATA_PATH + 'synthetic/LFR/', 'dataset_name': 'LFR-t1=3;t2=1.5;mu=0.3;average_degree=5;'}),
        (SyntheticFormatter, Experiment, GBModel,
         False, {'path': DATA_PATH + 'synthetic/LFR/', 'dataset_name': 'LFR-t1=3;t2=1.5;mu=0.4;average_degree=5;'}),
        (SyntheticFormatter, Experiment, GBModel,
         False, {'path': DATA_PATH + 'synthetic/LFR/', 'dataset_name': 'LFR-t1=3;t2=1.5;mu=0.5;average_degree=5;'}),
        (SyntheticFormatter, Experiment, GBModel,
         False, {'path': DATA_PATH + 'synthetic/LFR/', 'dataset_name': 'LFR-t1=3;t2=1.5;mu=0.6;average_degree=5;'}),
        (SyntheticFormatter, Experiment, GBModel,
         False, {'path': DATA_PATH + 'synthetic/LFR/', 'dataset_name': 'LFR-t1=3;t2=1.5;mu=0.7;average_degree=5;'}),
        (SyntheticFormatter, Experiment, GBModel,
         False, {'path': DATA_PATH + 'synthetic/LFR/', 'dataset_name': 'LFR-t1=3;t2=1.5;mu=0.8;average_degree=5;'}),
        (SyntheticFormatter, Experiment, GBModel,
         False, {'path': DATA_PATH + 'synthetic/LFR/', 'dataset_name': 'LFR-t1=3;t2=1.5;mu=0.9;average_degree=5;'}),
        (SyntheticFormatter, Experiment, GBModel,
         False, {'path': DATA_PATH + 'synthetic/LFR/', 'dataset_name': 'LFR-t1=3;t2=1.5;mu=1;average_degree=5;'}),

        # 2. Real-word OLP
        (RealWorldNonAttrFormatter, Experiment, GBModel,
         False, {'dataset_name': 'Email_network_Uni_R-V_Spain_Email_network_Uni_R-V_Spain'}),
        (RealWorldNonAttrFormatter, Experiment, GBModel,
         False, {'dataset_name': '595b15bd549f067e0263b525'}),
        (RealWorldNonAttrFormatter, Experiment, GBModel,
         False, {'dataset_name': 'Malaria_var_DBLa_HVR_networks_HVR_networks_9'}),
    ]

    data = [
        # NEW EXPERIMENTS #
        # 1. Feature selection
        #(RealWorldAttrFormatter, FeatureSelectionExperiment, NNModel,
        # True, {'path': DATA_PATH + 'real_world_data/', 'dataset_name': 'citeseer'}),
        #(RealWorldAttrFormatter, FeatureSelectionExperiment, NNModel,
        # True, {'path': DATA_PATH + 'real_world_data/', 'dataset_name': 'cora_ml'})
        # 2. Synthetic attributed
        (SyntheticFormatter, Experiment, NNModel,
         True, {'path': DATA_PATH + 'synthetic/acMark/', 'dataset_name': 'acMark-a=0.1;b=0.1;s=0.1;o=0.1'}),
        (SyntheticFormatter, Experiment, NNModel,
         True, {'path': DATA_PATH + 'synthetic/acMark/', 'dataset_name': 'acMark-a=0.2;b=0.1;s=0.1;o=0.1'}),
        (SyntheticFormatter, Experiment, NNModel,
         True, {'path': DATA_PATH + 'synthetic/acMark/', 'dataset_name': 'acMark-a=0.3;b=0.1;s=0.1;o=0.1'}),
        (SyntheticFormatter, Experiment, NNModel,
         True, {'path': DATA_PATH + 'synthetic/acMark/', 'dataset_name': 'acMark-a=0.4;b=0.1;s=0.1;o=0.1'}),
        (SyntheticFormatter, Experiment, NNModel,
         True, {'path': DATA_PATH + 'synthetic/acMark/', 'dataset_name': 'acMark-a=0.5;b=0.1;s=0.1;o=0.1'}),
        (SyntheticFormatter, Experiment, NNModel,
         True, {'path': DATA_PATH + 'synthetic/acMark/', 'dataset_name': 'acMark-a=0.6;b=0.1;s=0.1;o=0.1'}),
        (SyntheticFormatter, Experiment, NNModel,
         True, {'path': DATA_PATH + 'synthetic/acMark/', 'dataset_name': 'acMark-a=0.7;b=0.1;s=0.1;o=0.1'}),
        (SyntheticFormatter, Experiment, NNModel,
         True, {'path': DATA_PATH + 'synthetic/acMark/', 'dataset_name': 'acMark-a=0.8;b=0.1;s=0.1;o=0.1'}),
        (SyntheticFormatter, Experiment, NNModel,
         True, {'path': DATA_PATH + 'synthetic/acMark/', 'dataset_name': 'acMark-a=0.9;b=0.1;s=0.1;o=0.1'}),
        (SyntheticFormatter, Experiment, NNModel,
         True, {'path': DATA_PATH + 'synthetic/acMark/', 'dataset_name': 'acMark-a=1;b=0.1;s=0.1;o=0.1'})
    ]

    data = [
        (RealWorldAttrFormatter, Experiment, NNModel,
         True, {'path': DATA_PATH + 'real_world_data/', 'dataset_name': 'citeseer'}),
        (RealWorldAttrFormatter, FeatureSelectionExperiment, NNModel,
         True, {'path': DATA_PATH + 'real_world_data/', 'dataset_name': 'citeseer'}),
    ]

    for i, (formatter, experiment, model, attributed, args) in enumerate(data):
        exp_type = 'Feature selection' if experiment == FeatureSelectionExperiment else 'Legacy'
        experiment_path = f"{args['dataset_name']}, attributed is {attributed}, model: {model.MODEL_TYPE}, exp: {exp_type}"

        if not os.path.isdir(RESULT_PATH + experiment_path + '/'):
            try:
                os.makedirs(RESULT_PATH + experiment_path + '/')

                log.info('STARTING EXPERIMENT')
                log.info(experiment_path)

                e = experiment(formatter(args, attributed), model)
                e.run(attributed, experiment_path)
            except BaseException as e:
                log.error(e)
                log.error("Failed to compute, try again later")

                if os.path.exists(experiment_path) and os.path.isdir(experiment_path):
                    shutil.rmtree(experiment_path)
        else:
            log.warning(f'The path\n{experiment_path}\nSeems to already exist, skipping...')


__main__()
