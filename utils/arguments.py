import argparse


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("options", nargs="*")
    return parser.parse_args()
