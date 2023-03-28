import pydantic
from fastapi_utils.inferring_router import InferringRouter
from fastapi_utils.cbv import cbv
from fastapi.responses import JSONResponse
from fastapi import FastAPI

import uvicorn
import dill
from os.path import exists

from main import build_controller
from dte_stand.data_structures import Flows, Flow
from dte_stand.hash_function.hash import HashFunction
from dte_stand.hash_function.dxhash import WeightedDxHashFunction
from dte_stand.hash_function.base import BaseHashFunction
from dte_stand.algorithm.mate.lib.run_experiment import Runner
from main import glob_var


class Config:
    arbitrary_types_allowed = True

@pydantic.dataclasses.dataclass(config=Config)
class Dataclass:
    hash_function: BaseHashFunction
    runner: Runner
    current_flows: Flows

app = FastAPI()
router = InferringRouter()

app.glob_var = 4

@cbv(router)
class RestController:
    def __init__(self, experiment_folder='data_examples'):
        self.experiment_controller = build_controller(experiment_folder)
        self.hash_weights = {}

    @router.get("/run_full_epoch/{time}/{iteration}")
    def run_full_epoch(self, time, iteration):
        current_flows = self.experiment_controller.input_data.flows.get(int(time))
        current_topo, current_time = self.experiment_controller._get_current_topology_and_time(int(time))
        weights = self.experiment_controller.algorithm.step(current_topo, current_flows, iteration)
        if not weights:
            return JSONResponse(
                status_code=404,
                content={"message": "Item not found"},
            )
        file = 'weights-' + str(time) + '.json'
        with open(file, 'wb') as fp:
            dill.dump(weights, fp)

    @router.get("/status/{time}")
    def status(self, time):
        app.glob_var += 6
        print("GLOB", app.glob_var)
        return exists('weights-' + str(time) + '.json')

    @router.get("/nexthop/{time}/{start_node}/{end_node}")
    def nexthop(self, time, start_node, end_node):
        hash_function = HashFunction(self.experiment_controller.path_calculator)
        with open('weights-' + str(time) + '.json', 'rb') as file:
            hash_weights = dill.load(file)
        current_flows = self.experiment_controller.input_data.flows.get(int(time))
        current_topo, current_time = self.experiment_controller._get_current_topology_and_time(int(time))
        for flow in current_flows:
            if flow.start == start_node and flow.end == end_node:
                return hash_function.run(current_topo, flow, hash_weights)
        return JSONResponse(
            status_code=404,
            content={"message": "Flow not found"},
        )

    @router.get("/run_new_episode/{time}")
    def run_new_episode(self, time):
        hash_function = WeightedDxHashFunction(self.experiment_controller.path_calculator)
        current_flows = self.experiment_controller.input_data.flows.get(int(time))
        current_topo, current_time = self.experiment_controller._get_current_topology_and_time(int(time))
        runner = Runner(current_topo, hash_function).agent
        runner.env.get_current_flows(current_flows)
        states, actions, rewards, log_probs, values, last_value, phi = runner.run_episode()


app.include_router(router)
if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8080)


