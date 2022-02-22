import logging as log
import os
import shutil
import ray

from experiments.var_feature_selection_experiment import VaryingFeatureSelectionExperiment
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
        (SyntheticFormatter, FeatureSelectionExperiment, NNModel,
         True, {'path': DATA_PATH + 'synthetic/acMark/', 'dataset_name': 'acMark-a=0.1;b=0.1;s=0.1;o=0.1_run1'}),
        (SyntheticFormatter, FeatureSelectionExperiment, NNModel,
         True, {'path': DATA_PATH + 'synthetic/acMark/', 'dataset_name': 'acMark-a=0.1;b=0.1;s=0.1;o=0.1_run2'}),
        (SyntheticFormatter, FeatureSelectionExperiment, NNModel,
         True, {'path': DATA_PATH + 'synthetic/acMark/', 'dataset_name': 'acMark-a=0.1;b=0.1;s=0.1;o=0.1_run3'}),

        (SyntheticFormatter, FeatureSelectionExperiment, NNModel,
         True, {'path': DATA_PATH + 'synthetic/acMark/', 'dataset_name': 'acMark-a=0.2;b=0.1;s=0.1;o=0.1_run1'}),
        (SyntheticFormatter, FeatureSelectionExperiment, NNModel,
         True, {'path': DATA_PATH + 'synthetic/acMark/', 'dataset_name': 'acMark-a=0.2;b=0.1;s=0.1;o=0.1_run2'}),
        (SyntheticFormatter, FeatureSelectionExperiment, NNModel,
         True, {'path': DATA_PATH + 'synthetic/acMark/', 'dataset_name': 'acMark-a=0.2;b=0.1;s=0.1;o=0.1_run3'}),

        (SyntheticFormatter, FeatureSelectionExperiment, NNModel,
         True, {'path': DATA_PATH + 'synthetic/acMark/', 'dataset_name': 'acMark-a=0.3;b=0.1;s=0.1;o=0.1_run1'}),
        (SyntheticFormatter, FeatureSelectionExperiment, NNModel,
         True, {'path': DATA_PATH + 'synthetic/acMark/', 'dataset_name': 'acMark-a=0.3;b=0.1;s=0.1;o=0.1_run2'}),
        (SyntheticFormatter, FeatureSelectionExperiment, NNModel,
         True, {'path': DATA_PATH + 'synthetic/acMark/', 'dataset_name': 'acMark-a=0.1;b=0.1;s=0.1;o=0.1_run3'}),
    ]

    exp_types = {
        Experiment: "Legacy",
        FeatureSelectionExperiment: "Feature selection",
        VaryingFeatureSelectionExperiment: "Varying feature selection"
    }

    for i, (formatter, experiment, model, attributed, args) in enumerate(data):
        exp_type = exp_types[experiment]
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
