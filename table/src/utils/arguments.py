import argparse


def get_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--config",
        required=True,
        default="../config/lgbm_0.yml",
        help="Config file path",
    )
    parser.add_argument("--log_dir", default="./log", help="Directory to save log")
    parser.add_argument(
        "--debug", action="store_true", help="Whether to use debug mode"
    )
    return parser


def get_preprocess_parser() -> argparse.ArgumentParser:
    parser = get_parser()
    parser.add_argument(
        "--force", action="store_true", help="Overwrite existing feature files"
    )
    parser.add_argument(
        "--dryrun",
        action="store_true",
        help="Use subset of train.csv to calculate the feature",
    )
    return parser


def get_making_seed_average_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser()
    parser.add_argument("--base_config", required=True, help="Base config file path")
    parser.add_argument("--num_seeds", required=True, help="num seeds")

    return parser


def get_seed_average_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", required=True, help="Base output file path")

    return parser
