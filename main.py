import datetime
import argparse
import yaml
from tqdm import tqdm

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.data import DataLoader
from datasets import FaceDataset, CascadeStage2Dataset
from torch.utils.tensorboard import SummaryWriter

# Custom modules
from utils import *
from criteria import *

from resnet import *

parser = argparse.ArgumentParser(description='Train a model')
parser.add_argument('--config', type=str, default='config/raw_CNN_noAug_MSE_lr0.3_B16.yaml', help='Path to config file')
args = parser.parse_args()

config = yaml.load(open(args.config, 'r'), Loader=yaml.FullLoader)

# Hyperparameters etc.
# DEVICE
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
print("Using device:", DEVICE)

TASK = config['TASK']

STAGE = config['CASCADE']['STAGE']
STAGE1_MODEL_NAME = config['CASCADE']['STAGE1_MODEL_NAME']
STAGE1_MODEL_PATH = config['CASCADE']['STAGE1_MODEL_PATH']
# Data
TRAIN_PATH = config['DATA']['TRAIN_PATH']
TRAIN_PATH2 = config['DATA']['TRAIN_PATH2']
TEST_PATH = config['DATA']['TEST_PATH']
TRAIN_AUGMENT = config['DATA']['TRAIN_AUGMENT']
# TRAIN_PARTIAL = config['DATA']['TRAIN_PARTIAL']
# Training hyperparameters
LEARNING_RATE = config['TRAIN']['LEARNING_RATE']
BATCH_SIZE = config['TRAIN']['BATCH_SIZE']
NUM_EPOCHS = config['TRAIN']['NUM_EPOCHS']
SAVE_MODEL = config['TRAIN']['SAVE_MODEL']
LOSS = config['TRAIN']['LOSS']
timestamp = datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
LOG_PATH = "logs/" + TASK + "/" + timestamp + ".log"
WRITER_PATH = "runs/" + TASK + "/" + timestamp

# Model hyperparameters
MODEL_NAME = config['MODEL']['MODEL_NAME']
NUM_OUTPUTS = config['MODEL']['NUM_OUTPUTS']
PRETRAINED = config['MODEL']['PRETRAINED']
LOAD_MODEL = config['MODEL']['LOAD_MODEL']
LOAD_PATH = config['MODEL']['LOAD_PATH']

log = get_logger(LOG_PATH)

# Log hyperparameters
log.info("Task: {}".format(TASK))

if STAGE == 1:
    log.info("Training cascade stage 1")
elif STAGE == 2:
    log.info("Training cascade stage 2")
    log.info("Stage 1 model: {}".format(STAGE1_MODEL_NAME))
    log.info("Stage 1 model path: {}".format(STAGE1_MODEL_PATH))
log.info("Using device: {}".format(DEVICE))
log.info("Using config: {}".format(args.config))
log.info("Train path: {}".format(TRAIN_PATH))
if STAGE == 1:
    log.info("Train path 2: {}".format(TRAIN_PATH2))
log.info("Test path: {}".format(TEST_PATH))
log.info("Train augment: {}".format(TRAIN_AUGMENT))

log.info("Learning rate: {}".format(LEARNING_RATE))
log.info("Batch size: {}".format(BATCH_SIZE))
log.info("Num epochs: {}".format(NUM_EPOCHS))
log.info("Save model: {}".format(SAVE_MODEL))
log.info("Loss: {}".format(LOSS))
log.info("Log path: {}".format(LOG_PATH))
log.info("Writer path: {}".format(WRITER_PATH))

log.info("Model name: {}".format(MODEL_NAME))
log.info("Num outputs: {}".format(NUM_OUTPUTS))
log.info("Pretrained: {}".format(PRETRAINED))
log.info("Load model: {}".format(LOAD_MODEL))
log.info("Load path: {}".format(LOAD_PATH))


# Load Data
log.info("Loading data...")

if STAGE == 1:
    log.info("Load dataset for cascade stage 1")
    train_dataset1 = FaceDataset(
        path=TRAIN_PATH, partial=True, augment=TRAIN_AUGMENT)
    train_dataset2 = FaceDataset(
        path=TRAIN_PATH2, partial=True, augment=TRAIN_AUGMENT)
    train_dataset = torch.utils.data.ConcatDataset(
        [train_dataset1, train_dataset2])
    test_dataset = FaceDataset(path=TEST_PATH, partial=True, augment=False)
elif STAGE == 2:
    log.info("Load stage 1 model")
    stage1_model = resnet18(pretrained=False, num_classes=10).to(DEVICE)
    load_checkpoint(torch.load(STAGE1_MODEL_PATH), stage1_model)
    log.info("Load dataset for cascade stage 2")
    train_dataset = CascadeStage2Dataset(
        path=TRAIN_PATH, model=stage1_model, augment=TRAIN_AUGMENT, device=DEVICE)
    test_dataset = CascadeStage2Dataset(
        path=TEST_PATH, model=stage1_model, augment=False, device=DEVICE)
else:
    log.info("Load dataset for {}".format(TASK))
    train_dataset = FaceDataset(
        path=TRAIN_PATH, partial=False, augment=TRAIN_AUGMENT)
    test_dataset = FaceDataset(path=TEST_PATH, partial=False, augment=False)

train_loader = DataLoader(dataset=train_dataset,
                          batch_size=BATCH_SIZE, shuffle=True)

test_loader = DataLoader(dataset=test_dataset,
                         batch_size=BATCH_SIZE, shuffle=True)

# Initialize network
log.info("Initializing network {} with {} outputs...".format(
    MODEL_NAME, NUM_OUTPUTS))
if MODEL_NAME == "resnet18":
    model = resnet18(pretrained=PRETRAINED, num_classes=NUM_OUTPUTS).to(DEVICE)
elif MODEL_NAME == "resnet34":
    model = resnet34(pretrained=PRETRAINED, num_classes=NUM_OUTPUTS).to(DEVICE)
elif MODEL_NAME == "resnet50":
    model = resnet50(pretrained=PRETRAINED, num_classes=NUM_OUTPUTS).to(DEVICE)
elif MODEL_NAME == "resnet101":
    model = resnet101(pretrained=PRETRAINED, num_classes=NUM_OUTPUTS).to(DEVICE)
elif MODEL_NAME == "resnet152":
    model = resnet152(pretrained=PRETRAINED, num_classes=NUM_OUTPUTS).to(DEVICE)
else:
    log.error("Model not supported")
    exit()

# Model summary
log.info("Network: {}".format(model))

# Loss and optimizer
log.info("Initializing loss and optimizer...")
log.info("Loss: {}".format(LOSS))
# Choose loss function
if LOSS == "MSE" or LOSS == "L2":
    criterion = nn.MSELoss().to(DEVICE)
elif LOSS == "L1":
    criterion = nn.L1Loss().to(DEVICE)
elif LOSS == "SmoothL1":
    criterion = nn.SmoothL1Loss().to(DEVICE)
elif LOSS == "Wing":
    criterion = WingLoss().to(DEVICE)
elif LOSS == "SoftWing":
    criterion = SoftWingLoss().to(DEVICE)
else:
    log.error("Loss function not supported")
    exit()

log.info("Optimizer: {}".format("Adam"))
optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)
scheduler = optim.lr_scheduler.ExponentialLR(optimizer, gamma=0.8) 

# Tensorboard to get nice loss plots
# Create writer
log.info("Initializing tensorboard writer at: {}".format(WRITER_PATH))
writer = SummaryWriter(f"{WRITER_PATH}/")

# Keep track of training steps
train_step = 0
test_step = 0

# Load model if LOAD_MODEL = True
if LOAD_MODEL:
    log.info("Loading model and optimizer from: {}".format(LOAD_PATH))
    load_checkpoint(torch.load(LOAD_PATH), model, optimizer)

# Train Network
log.info("Training network...")
pre_test_nme = 1000
for epoch in range(NUM_EPOCHS):
    # Train the network
    train_loop = tqdm(train_loader, leave=True)
    train_loop.set_description(f"Train [{epoch+1}/{NUM_EPOCHS}]")
    train_step, train_nme = train(
        train_loop, model, optimizer, criterion, writer, train_step, DEVICE
    )
    log.info(f"EPOCH [{epoch+1}/{NUM_EPOCHS}] Train NME: {train_nme:.5f}")

    # Test the network
    test_loop = tqdm(test_loader, leave=True)
    test_loop.set_description(f"Test [{epoch+1}/{NUM_EPOCHS}]")
    test_step, test_nme = test(
        test_loop, model, criterion, writer, test_step, DEVICE
    )
    log.info(f"EPOCH [{epoch+1}/{NUM_EPOCHS}] Test NME: {test_nme:.5f}")
    log.info(f"EPOCH [{epoch+1}/{NUM_EPOCHS}] Learning rate: {optimizer.param_groups[0]['lr']:.5f}")

    # Save to tensorboard
    writer.add_scalar("Learning rate", optimizer.param_groups[0]["lr"], global_step=(epoch+1))
    writer.add_scalar("Train NME", train_nme, global_step=(epoch+1))
    writer.add_scalar("Test NME", test_nme, global_step=(epoch+1))
    scheduler.step()

    # Save model if performance improved
    if SAVE_MODEL and test_nme < pre_test_nme:
        checkpoint = {
            "state_dict": model.state_dict(),
            "optimizer": optimizer.state_dict(),
        }
        timestamp = datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
        log.info(f"EPOCH [{epoch+1}/{NUM_EPOCHS}] NME improved from {pre_test_nme:.5f} to {test_nme:.5f}")
        log.info(f"EPOCH [{epoch+1}/{NUM_EPOCHS}] Saving model to: checkpoints/{TASK}/{timestamp}_epoch_{epoch+1}_NME_{test_nme:.5f}.pth.tar")
        save_checkpoint(
            checkpoint, filename=f"checkpoints/{TASK}/{timestamp}_epoch_{epoch+1}_NME_{test_nme:.5f}.pth.tar")
        pre_test_nme = test_nme


# run code:
# python main.py --config config\raw_CNN_noAug_MSE\raw_CNN_noAug_MSE_lr0.6_B4.yaml

# for activate tensorboard:
# tensorboard --logdir=runs
