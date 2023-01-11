from selene_sdk.utils import load_path
from selene_sdk.utils import parse_configs_and_run
import argparse


parser = argparse.ArgumentParser(description='Train model with Selene.')
parser.add_argument('yml_file', type=str, help='Input file in yml format')
args = parser.parse_args()

configs = load_path(args.yml_file)
parse_configs_and_run(configs, lr=1e-05)
