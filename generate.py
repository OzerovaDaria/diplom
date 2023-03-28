from generator.generator import UniformFlowGenerator
from generator.demand_generator import DemandMatrixGenerator
from generator.parsers import sndlib_brain as sndlib_brain_parser
import json
import networkx
import sys


def generate_from_dataset():
    # create generator and set its parameters
    generator = UniformFlowGenerator(5, 400000, 2000000)

    # parse folder that contains dataset files
    matrices = sndlib_brain_parser.parse_all('generator/dataset/')

    # run the generator
    result = generator.generate(matrices, 60000)

    # convert results to a string with optional pretty print - can be removed, only result.json() is needed
    # But results will be unreadable for a human is pretty print is removed
    pretty_res = json.dumps(json.loads(result.json()), indent=4)

    # open result file and write the flow data
    with open('flows.log', 'w') as f:
        f.write(pretty_res)


def generate_synthetic(mode: str, seed: int = None):
    # get a topology
    with open('data_examples/rhombus/topology.gml', mode='rb') as f:
        topology = networkx.readwrite.read_gml(f)

    # create generator
    if mode == 'standard':
        # min_bw_coef, max_bw_coef = 0.35, 0.5 # 90%
        # min_bw_coef, max_bw_coef = 0.2, 0.3 # 80%
        # min_bw_coef, max_bw_coef = 0.12, 0.21 # 70%
        # min_bw_coef, max_bw_coef = 0.09, 0.16 # 60%
        # min_bw_coef, max_bw_coef = 0.06, 0.12 # 50%
        # min_bw_coef, max_bw_coef = 0.05, 0.085 # 40%
        min_bw_coef, max_bw_coef = 0.03, 0.06 # 30%
        # min_bw_coef, max_bw_coef = 0.02, 0.04 # 20%
        # min_bw_coef, max_bw_coef = 0.008, 0.02 # 10%
    else: # 'gravity' or any other string => gravity mode
        bw_variation_sqrt = 3.5
        coef = bw_variation_sqrt * 1.3 # may need to be changed manually when bw_variation_sqrt is changed
        # mean_bw_expected = 1.10 / coef # 90%
        # mean_bw_expected = 0.70 / coef # 80%
        # mean_bw_expected = 0.57 / coef # 70%
        mean_bw_expected = 0.48 / coef # 60%
        # mean_bw_expected = 0.40 / coef # 50%
        # mean_bw_expected = 0.32 / coef # 40%
        # mean_bw_expected = 0.24 / coef # 30%
        # mean_bw_expected = 0.16 / coef # 20%
        # mean_bw_expected = 0.08 / coef # 10%

        # bw_variation = 0.02
        # min_bw_coef = max(0, mean_bw_expected - bw_variation)
        # max_bw_coef = mean_bw_expected + bw_variation

        min_bw_coef = mean_bw_expected / bw_variation_sqrt
        max_bw_coef = mean_bw_expected * bw_variation_sqrt
    generator = DemandMatrixGenerator(min_bw_coef, max_bw_coef, topology, mode=mode, seed=seed)

    # generate some matrices
    matrices = generator.generate(50)

    # generate flows using uniform flow generator
    flow_generator = UniformFlowGenerator(5, 20000, 500000)

    # run the generator
    result = flow_generator.generate(matrices, 30000)

    # convert results to a string with optional pretty print - can be removed, only result.json() is needed
    # But results will be unreadable for a human is pretty print is removed
    pretty_res = json.dumps(json.loads(result.json()), indent=4)

    # open result file and write the flow data
    with open('flows.json', 'w') as f:
        f.write(pretty_res)

def get_arg(i: int) -> str:
    """
    :return: sys.argv[i] if it's present, None otherwise
    """
    return sys.argv[i] if len(sys.argv) > i else None

if __name__ == '__main__':
    if get_arg(1) == 'help':
        print('Usage: generate.py [mode] [seed]\n'
              'mode: gravity (default), standard - is algorithm of synthetic demand generation:\n'
              'seed: random seed for synthetic demand generation')
        exit(0)
    mode = get_arg(1) or 'gravity' # "standard" or "gravity". default: "gravity"
    seed = get_arg(2) or None
    generate_synthetic(mode, seed=seed)