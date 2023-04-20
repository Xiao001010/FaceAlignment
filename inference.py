import argparse
import yaml
from tqdm import tqdm

import numpy as np
import torch
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt

import torchlm

from utils import *
from utils import Recover
from resnet import *
from datasets import FaceDataset, CascadeStage2Dataset

arg = argparse.ArgumentParser()
arg.add_argument('--config', type=str, default='config/inference.yaml', help='Path to config file')
args = arg.parse_args()

config = yaml.load(open(args.config, 'r'), Loader=yaml.FullLoader)

# Hyperparameters etc.

# DEVICE
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
DEVICE = "cpu"
print("Using device:", DEVICE)

TASK = config['TASK']
STAGE = config['CASCADE']['STAGE']
STAGE1_MODEL_NAME = config['CASCADE']['STAGE1_MODEL_NAME']
STAGE1_MODEL_PATH = config['CASCADE']['STAGE1_MODEL_PATH']
# Data
TEST_PATH = config['DATA']['TEST_PATH']
AUGMENT = config['DATA']['AUGMENT']
INFERENCE = config['DATA']['INFERENCE']

# Model hyperparameters
MODEL_NAME = config['MODEL']['MODEL_NAME']
NUM_OUTPUTS = config['MODEL']['NUM_OUTPUTS']
LOAD_PATH = config['MODEL']['LOAD_PATH']

if STAGE == 1:
    print("Load dataset for cascade stage 1")
    test_dataset = FaceDataset(path=TEST_PATH, partial=True, augment=False, inference=INFERENCE)
elif STAGE == 2:
    print("Load stage 1 model")
    stage1_model = resnet18(pretrained=False, num_classes=10).to(DEVICE)
    load_checkpoint(torch.load(STAGE1_MODEL_PATH), stage1_model)
    print("Load dataset for cascade stage 2")
    test_dataset = FaceDataset(path=TEST_PATH, partial=False, augment=False, inference=INFERENCE)
else:
    print("Load dataset for {}".format(TASK))
    test_dataset = FaceDataset(path=TEST_PATH, partial=False, augment=False, inference=INFERENCE)


test_loader = DataLoader(dataset=test_dataset,
                         batch_size=1, shuffle=False)

# Initialize network
print("Initializing network {} with {} outputs...".format(
    MODEL_NAME, NUM_OUTPUTS))
if MODEL_NAME == "resnet18":
    model = resnet18(pretrained=False, num_classes=NUM_OUTPUTS).to(DEVICE)
elif MODEL_NAME == "resnet34":
    model = resnet34(pretrained=False, num_classes=NUM_OUTPUTS).to(DEVICE)
elif MODEL_NAME == "resnet50":
    model = resnet50(pretrained=False, num_classes=NUM_OUTPUTS).to(DEVICE)
elif MODEL_NAME == "resnet101":
    model = resnet101(pretrained=False, num_classes=NUM_OUTPUTS).to(DEVICE)
elif MODEL_NAME == "resnet152":
    model = resnet152(pretrained=False, num_classes=NUM_OUTPUTS).to(DEVICE)
else:
    print("Model not supported")
    exit()

# Load model
print("Loading model from {}".format(LOAD_PATH))
load_checkpoint(torch.load(LOAD_PATH), model)

for i, (image, landmarks) in enumerate(tqdm(test_loader)):
    image = image.to(DEVICE)
    fake_image = torch.zeros_like(image)
    landmarks = landmarks.reshape(-1, 2)
    output = model(image).cpu().detach()
    image = image.cpu().detach()
    image, pred = Recover(image, output)
    fake_image, landmarks = Recover(fake_image, landmarks)
    plt.imshow(image)
    plt.plot(pred[:, 0], pred[:, 1], 'bo')
    plt.plot(landmarks[:, 0], landmarks[:, 1], 'ro')
    plt.show()
    break
