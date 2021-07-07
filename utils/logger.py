import os

import mlcrate as mlc
from loguru import logger


def get_logger(config, exp_num):
    logger.remove()
    logger.add(os.path.join(config.log_path, f"exp_{exp_num}.log"))
    header = config.header.split(" ")
    csv_log = mlc.LinewiseCSVWriter(
        os.path.join(config.log_path, f"exp_{exp_num}.csv"), header=header
    )
    return logger, csv_log
