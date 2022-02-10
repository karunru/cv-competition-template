import os
from glob import glob
from pathlib import Path

import pandas as pd


def find_exp_num(config_path: str) -> int:

    config_files = glob(str(Path(config_path) / "exp_*.yaml"))
    if not len(config_files):
        return 1
    else:
        exp_nums = [
            os.path.splitext(i)[0].split("/")[-1].split("_")[-1] for i in config_files
        ]
        exp_nums = list(map(int, exp_nums))
        return max(exp_nums) + 1


def remove_abnormal_exp(log_path: str, config_path: str) -> None:
    log_files = glob(str(Path(log_path) / "*.csv"))
    for log_file in log_files:
        log_df = pd.read_csv(log_file)
        if len(log_df) == 0:
            exp_num = os.path.splitext(log_file)[0].split("/")[-1].split("_")[-1]
            os.remove(os.path.join(log_path, f"exp_{exp_num}.log"))
            os.remove(os.path.join(log_path, f"exp_{exp_num}.csv"))
            os.remove(os.path.join(config_path, f"exp_{exp_num}.yaml"))
