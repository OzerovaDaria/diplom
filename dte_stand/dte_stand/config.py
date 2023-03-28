import yaml
import os
from typing import Optional
from pydantic import BaseModel, PositiveInt


class YamlConfig(BaseModel):
    hash_function: str
    algorithm: str
    path_calculator: str
    iterations: PositiveInt
    lsdb_period: PositiveInt
    log_path: str
    log_level: str
    debug_check_cycles: int


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
