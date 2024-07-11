import yaml
from easydict import EasyDict

from raptor import RaptorHandler


def read_yaml_config(file_path):
    with open(file_path, "r") as file:
        config_data = yaml.safe_load(file)
    return EasyDict(config_data)


config = read_yaml_config("./config.yaml")
handler = RaptorHandler(config)

handler.build(raw_data_path="./data")
print("======build done==========")
response = handler.ask("什么样的数据算是个人数据")
print(response)
