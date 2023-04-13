import argparse
from tqdm import tqdm

import torch
# import torch.nn as nn

from resnet import *

arg = argparse.ArgumentParser()
arg.add_argument('--config', type=str, default='config/inference.yaml', help='Path to config file')
args = arg.parse_args()



