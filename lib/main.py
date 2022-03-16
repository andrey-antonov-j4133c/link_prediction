import logging as log
import os
import shutil
import ray
from multiprocessing import Process

from experiments.var_feature_selection_experiment import VaryingFeatureSelectionExperiment
from models.nn_model import NNModel
from models.gb_model import GBModel

from settings import *

from data_formatting.real_world_attributed_networks import RealWorldAttrFormatter
from data_formatting.synthetic_networks import SyntheticFormatter
from data_formatting.real_world_non_attributed_networks import RealWorldNonAttrFormatter

from experiments.experiment import Experiment
from experiments.feature_selection_experiment import FeatureSelectionExperiment


def run_experiments():
    ray.init()

    if VEBROSE:
        log.basicConfig(format="%(levelname)s: %(message)s", level=log.INFO)
        log.info("Verbose output.")
    else:
        log.basicConfig(format="%(levelname)s: %(message)s")

    data = []

    for a in ('0.1', '0.2', '0.3', '0.4', '0.5', '0.6', '0.7', '0.8', '0.9', '1'):
        for run in (1, 2, 3, 4, 5):
            data.append(
                (SyntheticFormatter, Experiment, NNModel,
                 True, {'path': DATA_PATH + 'synthetic/acMark/', 'dataset_name': f'acMark-a={a};b=0.1;s=0.1;o=0.1_run{run}'})
            )

    data.append(
       (SyntheticFormatter, VaryingFeatureSelectionExperiment, NNModel,
        True, {'path': DATA_PATH + 'synthetic/acMark/', 'dataset_name': f'acMark-a=0.5;b=0.1;s=0.1;o=0.1_run1'})
    )

    data.append(
        (RealWorldNonAttrFormatter, FeatureSelectionExperiment, NNModel,
         True, {'path': DATA_PATH + 'real_world_data/OLP/', 'dataset_name': f'595b15bd549f067e0263b525'})
    )

    data.append(
        (RealWorldAttrFormatter, FeatureSelectionExperiment, NNModel,
         True, {'path': DATA_PATH + 'real_world_data/', 'dataset_name': f'citeseer.npz'})
    )

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


def main():
    t = Process(target=run_experiments)
    t.start()
    while True:
        choice = input()
        if choice == "s":
            t.terminate()
            ray.shutdown()
            raise SystemExit


if __name__ == '__main__':
    main()
