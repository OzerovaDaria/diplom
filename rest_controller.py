import uuid
from fastapi_utils.inferring_router import InferringRouter
from fastapi_utils.cbv import cbv
from fastapi import FastAPI

from typing import List
import uvicorn
from pydantic import BaseSettings
from networkx import MultiDiGraph

from main import build_controller
from dte_stand.controller import ExperimentController
from dte_stand.data_structures.hash_weights import HashWeights
from dte_stand.data_structures.flows import Flow
from dte_stand.hash_function.dxhash import WeightedDxHashFunction
from dte_stand.algorithm.mate.lib.run_experiment import Runner

class Settings(BaseSettings):
    current_time: int = 0
    experiment_controller: ExperimentController = build_controller(experiment_folder='data_examples')
    current_topo: MultiDiGraph = experiment_controller._get_current_topology_and_time(current_time)[0]
    current_flows: List[Flow] = experiment_controller.input_data.flows.get(current_time)
    hash_weights: HashWeights = HashWeights()
    hash_function: WeightedDxHashFunction = WeightedDxHashFunction(experiment_controller.path_calculator)
    flow_ids: dict = {}

settings = Settings()

app = FastAPI()
app.runner = Runner(settings.current_topo, settings.hash_function, False, settings.current_flows).agent
router = InferringRouter()

@cbv(router)
class RestController:
    @router.get("/run_new_episode")
    def run_new_episode(self):
        states, actions, rewards, log_probs, values, last_value, phi = app.runner.run_episode()
        return phi

    @router.get("/nexthop/{current_node}/{end_node}/{prev_node}/{dest_ip}/{source_ip}/{dest_port}")
    def nexthop(self, current_node, end_node, prev_node, dest_ip, source_ip, dest_port):
        if dest_ip + source_ip + dest_port not in settings.flow_ids.keys():
            settings.flow_ids[dest_ip + source_ip + dest_port] = str(uuid.uuid4())
        return settings.hash_function._flow_path(settings.current_topo, Flow(start=prev_node, end=end_node, 
                                all_bandwidth={}, start_time=0, end_time=1, bandwidth=100, flow_id=settings.flow_ids[dest_ip + source_ip + dest_port]),
                                settings.hash_weights, current_node, depth=1)

    @router.get("/run_new_epoch/{time}")
    def run_new_epoch(self, time):
        settings.current_topo, settings.current_time = settings.experiment_controller._get_current_topology_and_time(int(time))
        settings.current_flows = settings.experiment_controller.input_data.flows.get(settings.current_time)
        app.runner = Runner(settings.current_topo, settings.hash_function, True, settings.current_flows).agent

app.include_router(router)
if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8080)


