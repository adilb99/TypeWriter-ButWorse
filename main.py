import os
import sys
import json
import time
import argparse
from easydict import EasyDict

from train import TWAgent

root_dir = os.path.realpath(__file__)
while not root_dir.endswith("TYPEWRITER-BUTWORSE"):
  root_dir = os.path.dirname(root_dir)
sys.path.append(root_dir)


def parse_arguments():
    parser = argparse.ArgumentParser()
    parser.add_argument('-mode', type=str,
                         default="train", help="tarining or testing")
    parser.add_argument('-config', type=str,
                         default="./config.json", 
                         help="path to the config file")

    args = parser.parse_args()
    return args


def read_cfg(cfg_file):
    cfg = json.loads(cfg_file)
    cfg = EasyDict(cfg)
    return cfg


def main():
    args = parse_arguments()
    cfg = read_cfg(args.config)
    agent = TWAgent(cfg)

    if args.m == 'train':
        return 
        agent.train()
