import importlib
import sys
from dte_stand.config import Config
from dte_stand.logger import init_logger
from dte_stand.controller import ExperimentController

glob_var = 8

def dynamic_import(object_path: str, **module_kwargs):
    module, object = object_path.rsplit('.', 1)
    imported_module = importlib.import_module(module)
    obj_class = getattr(imported_module, object)
    return obj_class(**module_kwargs)

def build_controller(experiment_folder):
    Config.load_config(experiment_folder)
    config = Config.config()

    init_logger(config.log_path, config.log_level, [])

    path_calculator = dynamic_import(config.path_calculator)
    hash_function = dynamic_import(config.hash_function, path_calculator=path_calculator,
                                   debug_check_cycles=config.debug_check_cycles)
    algo = dynamic_import(config.algorithm, hash_function=hash_function)

    controller = ExperimentController(
            experiment_folder, config.lsdb_period, config.iterations, hash_function, algo, path_calculator)
    return controller

if __name__ == '__main__':
    try:
        experiment_folder = sys.argv[1]
    except IndexError:
        print('Usage: main.py path_to_folder\n'
              'path_to_folder is a path to folder that contains experiment input data\n'
              'data_examples in this repository is an example of the structure of this folder\n')
        exit(1)
    phi = build_controller(experiment_folder).run()
