import sys

import pandas as pd
import yaml

if __name__ == "__main__":

    sys.path.append("./")
    from table.src.utils import tools

    config_path = "input/convert_config.yml"
    with open(config_path, "r") as f:
        config = dict(yaml.load(f, Loader=yaml.SafeLoader))

    target = config["target"]
    input_extension = config["input_extension"]
    output_extension = config["output_extension"]
    input_dir = config["input_dir"]
    output_dir = config["output_dir"]

    for t in target:
        print("convert ", t)
        (
            tools.reduce_mem_usage(
                pd.read_csv(input_dir + t + "." + input_extension, encoding="utf-8")
            )
        ).to_pickle(output_dir + t + "." + output_extension)
