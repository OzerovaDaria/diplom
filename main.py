import importlib
import sys
import os
import argparse
from datetime import datetime
from dte_stand.config import Config
from dte_stand.history import HistoryTracker
from dte_stand.logger import init_logger
from dte_stand.phi_calculator import PhiCalculator
from dte_stand.controller import ExperimentController, RandomExperimentController


def dynamic_import_function(object_path):
    module, object, function_name = object_path.rsplit('.', 2)
    imported_module = importlib.import_module(module)
    obj_class = getattr(imported_module, object)
    return getattr(obj_class, function_name)


def dynamic_import(object_path: str, **module_kwargs):
    module, object = object_path.rsplit('.', 1)
    imported_module = importlib.import_module(module)
    obj_class = getattr(imported_module, object)
    return obj_class(**module_kwargs)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('experiment_folder', type=str,
                        help='path to folder that contains experiment input data\n'
                             'data_examples in this repository is an example of the structure of this folder')
    parser.add_argument('-n', '--name', type=str, default='',
                        help='description of the experiment')
    parser.add_argument('-m', '--model', type=str, default=None,
                        help='path to folder of saved model to load into current experiment')
    args = parser.parse_args()

    Config.load_config(args.experiment_folder)
    config = Config.config()

    folder_name = datetime.now().strftime('%H-%M-%S,%d-%m-%Y')
    result_path = os.path.join(args.experiment_folder, folder_name + '_' + args.name)
    os.mkdir(result_path)
    PhiCalculator.set_plot_folder(result_path)
    HistoryTracker.set_result_folder(result_path)

    init_logger(os.path.join(result_path, config.log_path), config.log_level, ['matplotlib'])

    '''
    print("CONFIG PHI", config.phi, "config.path_calculator", config.path_calculator)
    print("CONFIG HASH", config.hash_function, "CONFIG DEG", config.debug_check_cycles)
    print("CONFIG ALG", config.algorithm, type(config.algorithm), result_path)
    print("MODEL", args.model, type(args.model))
    '''
    print("RESULT PATH", type(result_path), result_path)
    print("MODEL DIR", type(args.model), args.model)
    
    phi_func = dynamic_import_function(config.phi)
    path_calculator = dynamic_import(config.path_calculator)
    hash_function = dynamic_import(config.hash_function, path_calculator=path_calculator,
                                   debug_check_cycles=config.debug_check_cycles)
    algo = dynamic_import(config.algorithm, hash_function=hash_function, phi_func=phi_func, experiment_dir=result_path,
                          model_dir=args.model)

    controller = ExperimentController(
            args.experiment_folder, config.lsdb_period, config.iterations, hash_function,
            algo, path_calculator, phi_func, result_path)
    phi = controller.run()

    PhiCalculator.plot_full(all_iterations=True)
