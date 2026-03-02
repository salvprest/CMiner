import os
import json
from importlib.resources import files


class NetworkConfigurator:
    def __init__(self, network_path, network_type):
        self.network_path = network_path
        self.network_type = network_type
        self.config       = self.configuration_reading()

    def configuration_reading(self):
        config_filename = self.network_type.lower() + "_conf.json"
        config_ref = files("NetworkLoader.configs").joinpath(config_filename)
        with config_ref.open('r') as file:
            config = json.load(file)
            # split path in file_path and file_name
            config["file_path"], config["file_name"] = os.path.split(self.network_path)
            return config