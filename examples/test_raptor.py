import yaml
from easydict import EasyDict

from original_raptor import RaptorHandler


def read_yaml_config(file_path):
    with open(file_path, "r") as file:
        config_data = yaml.safe_load(file)
    return EasyDict(config_data)


save_path = "/tmp/raptor_paper"
config = read_yaml_config("./config.yaml")
handler = RaptorHandler(config, save_path)

handler.build(raw_data_path="data/raptor_paper.pdf")
print("======build done==========")
response = handler.ask("give me the summary of the raptor paper")
print(response)
