import json
import argparse
from trainer_3DShape import train

from utils import parser

def main():
    args = setup_parser().parse_args()
    param = load_json(args.config)
    args = vars(args) # Converting argparse Namespace to a dict.
    args.update(param) # Add parameters from json
    args2 = parser.get_args() # add parameters for backbones
    args.update(vars(args2))
    train(args)

def load_json(setting_path):
    with open(setting_path) as data_file:
        param = json.load(data_file)
    return param

def setup_parser():
    parser = argparse.ArgumentParser(description='Reproduce of multiple pre-trained incremental learning algorthms.')
    parser.add_argument('--config', type=str, default='./exps/cil_config.json',
                        help='Json file of settings.')
    return parser

if __name__ == '__main__':
    main()
