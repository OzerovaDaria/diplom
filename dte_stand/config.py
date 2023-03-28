import yaml
import os
from typing import Optional
from pydantic import BaseModel, PositiveInt


class ActionDescription(BaseModel):
    action: bool
    value: float


class MateActions(BaseModel):
    addition: ActionDescription
    subtraction: ActionDescription
    multiplication: ActionDescription
    division: ActionDescription
    zero: ActionDescription


class MateConfig(BaseModel):
    horizons: int
    episodes: int
    gamma: float
    gae_lambda: float
    reward: str
    reward_computation: str
    min_weight: float
    max_weight: float
    greedy_epsilon: float
    actions: MateActions
    message_iterations: int = -1


class YamlConfig(BaseModel):
    hash_function: str
    algorithm: str
    path_calculator: str
    phi: str
    iterations: PositiveInt
    plot_period: int
    lsdb_period: PositiveInt
    log_path: str
    log_level: str
    debug_check_cycles: int
    store_hashweights: int
    store_nexthops: int
    mate: MateConfig


class Config:
    _config: Optional[YamlConfig] = None

    @classmethod
    def load_config(cls, path_to_folder: str) -> None:
        with open(os.path.join(path_to_folder, 'config.yaml'), 'r') as f:
            config_dict = yaml.load(f, Loader=yaml.Loader)

        cls._config = YamlConfig.parse_obj(config_dict)

    @classmethod
    def config(cls) -> Optional[YamlConfig]:
        return cls._config
