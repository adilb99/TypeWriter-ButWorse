import os
import sys
import json
import time
import argparse
from easydict import EasyDict

from train import TWAgent


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
    with open(cfg_file) as f:
      cfg = json.loads(f.read())
    cfg = EasyDict(cfg)
    return cfg


def main():
    args = parse_arguments()
    cfg = read_cfg(args.config)
    agent = TWAgent(cfg)
    if args.mode == 'train':
        agent.train()

if __name__ == "__main__":
  main()