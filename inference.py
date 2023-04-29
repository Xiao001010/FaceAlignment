import argparse
import yaml
from tqdm import tqdm

import numpy as np
import torch
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt

from utils import *
from resnet import *
from datasets import FaceDataset, CascadeStage2Dataset

# load config file
arg = argparse.ArgumentParser()
arg.add_argument('--config', type=str, default='config/Inference/Inference_Cas_Stage2_noAug_MSE-S1_noAug_MSE-lr0.5g0.9_B2.yaml', help='Path to config file')
args = arg.parse_args()

config = yaml.load(open(args.config, 'r'), Loader=yaml.FullLoader)

# Hyperparameters etc.

# DEVICE
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
# DEVICE = "cpu"
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

# Inference hyperparameters
RECOVER = config['INFERENCE']['RECOVER']
PLOT_ON_RAW_DATA = config['INFERENCE']['PLOT_ON_RAW_DATA']
STOP_IDX = config['INFERENCE']['STOP_IDX']
SAVE_PREDICTIONS = config['INFERENCE']['SAVE_PREDICTIONS']
PLOT = config['INFERENCE']['PLOT']

# Load dataset
if STAGE == 1:
    print("Load dataset for cascade stage 1 from {}".format(TEST_PATH))
    test_dataset = FaceDataset(path=TEST_PATH, partial=True, augment=False, inference=INFERENCE)
elif STAGE == 2:
    print("Load stage 1 model")
    stage1_model = resnet18(pretrained=False, num_classes=10).to(DEVICE)
    load_checkpoint(torch.load(STAGE1_MODEL_PATH), stage1_model)
    print("Load dataset for cascade stage 2 from {}".format(TEST_PATH))
    if INFERENCE:
        test_dataset = CascadeStage2Dataset(path=TEST_PATH, model=stage1_model, augment=False, device=DEVICE, inference=INFERENCE)
    else:
        test_dataset = CascadeStage2Dataset(path=TEST_PATH, model=stage1_model, augment=False, device=DEVICE, inference=INFERENCE, test=True)
else:
    print("Load dataset for {} from {}".format(TASK, TEST_PATH))
    test_dataset = FaceDataset(path=TEST_PATH, partial=False, augment=False, inference=INFERENCE)

test_loader = DataLoader(dataset=test_dataset,
                         batch_size=1, shuffle=False)

# Load raw data for plotting
raw_data = np.load(TEST_PATH)
raw_images = raw_data['images']
try:
    raw_landmarks = raw_data['points']
except:
    print("No landmarks in raw data")
    raw_landmarks = None
raw_image = None
raw_landmark = None

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

# Inference
print("Starting inference...")
model.eval()
if STAGE != 2:
    print("Inference on test data...")
    indices = (20, 29, 16, 32, 38)
    if INFERENCE:
        for i, (image, fake_landmarks) in enumerate(tqdm(test_loader)):
            if PLOT_ON_RAW_DATA and RECOVER:
                raw_image = raw_images[i]
                if STAGE == 1:
                    try:
                        raw_landmark = raw_landmarks[i][indices, :]
                    except:
                        raw_landmark = None
                else:
                    try:
                        raw_landmark = raw_landmarks[i]
                    except:
                        raw_landmark = None
            image, pred = Inferencer(model, image, recover=RECOVER, landmarks=None, angle=0, 
                       device=DEVICE, raw_image=raw_image, raw_landmark=raw_landmark, plot=PLOT)
            if STOP_IDX and i == STOP_IDX:
                break
    else:
        for i, (image, landmarks) in enumerate(tqdm(test_loader)):
            if PLOT_ON_RAW_DATA and RECOVER:
                raw_image = raw_images[i]
                if STAGE == 1:
                    try:
                        raw_landmark = raw_landmarks[i][indices, :]
                    except:
                        raw_landmark = None
                else:
                    try:
                        raw_landmark = raw_landmarks[i]
                    except:
                        raw_landmark = None
            image, pred = Inferencer(model, image, recover=RECOVER, landmarks=landmarks, 
                       angle=0, device=DEVICE, raw_image=raw_image, raw_landmark=raw_landmark, plot=PLOT)
            if STOP_IDX and i == STOP_IDX:
                break
elif STAGE == 2:
    if INFERENCE:
        predictions =  []
        for i, (image, angle) in enumerate(tqdm(test_loader)):
            if PLOT_ON_RAW_DATA and RECOVER:
                raw_image = raw_images[i]
                try:
                    raw_landmark = raw_landmarks[i]
                except:
                    print("No landmarks in raw data")
                    raw_landmark = None
            angle = angle.item()
            image, pred = Inferencer(model, image, recover=RECOVER, landmarks=None, 
                       angle=angle, device=DEVICE, raw_image=raw_image, raw_landmark=raw_landmark, plot=PLOT)
            predictions.append(pred)
            if STOP_IDX and i == STOP_IDX:
                break
    else:
        for i, (image, landmarks, angle) in enumerate(tqdm(test_loader)):
            if PLOT_ON_RAW_DATA and RECOVER:
                # print("plot on raw data")
                raw_image = raw_images[i]
                raw_landmark = raw_landmarks[i]
            angle = angle.item()
            # print("angle", type(angle), angle)
            # print("landmark", max(raw_landmark[:, 0]), max(raw_landmark[:, 1]))
            image, pred = Inferencer(model, image, recover=RECOVER, landmarks=landmarks, 
                       angle=angle, device=DEVICE, raw_image=raw_image, raw_landmark=raw_landmark, plot=PLOT)
            if STOP_IDX and i == STOP_IDX:
                break

# Save predictions
if SAVE_PREDICTIONS and STAGE == 2 and INFERENCE and STOP_IDX is None:
    print("Saving predictions...")
    predictions = np.array(predictions)
    save_as_csv(predictions, "./results")