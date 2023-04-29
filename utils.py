from tqdm import tqdm
import datetime
import os
# import yaml

import cv2
import numpy as np
import torch
from torch.utils.tensorboard import SummaryWriter
import matplotlib.pyplot as plt

import logging

import torchlm

# list of all functions
__all__ = ["get_logger", 
           "save_checkpoint", 
           "load_checkpoint", 
           "NME", 
           "train", 
           "test", 
           "Recover", 
           "Inferencer", 
           "save_as_csv"]

def get_logger(path):
    """Get logger for logging

    Parameters
    ----------
    path : str
        path to log file

    Returns
    -------
    log : logging.Logger
        logger object
    """    
    if not os.path.exists(os.path.dirname(path)):
        print("Creating log directory {}".format(os.path.dirname(path)))
        os.makedirs(os.path.dirname(path))
    if not os.path.exists(path):
        print("Creating log file{}".format(path))
        open(path, 'a').close()

    log = logging.getLogger()
    log.setLevel(logging.INFO)
    formatter = logging.Formatter(fmt="[%(asctime)s|%(filename)s|%(levelname)s] %(message)s", 
                                  datefmt="%a %b %d %H:%M:%S %Y")
    ch = logging.StreamHandler()
    ch.setLevel(logging.INFO)
    ch.setFormatter(formatter)
    log.addHandler(ch)
    
    fh = logging.FileHandler(path)
    fh.setLevel(logging.INFO)
    fh.setFormatter(formatter)
    log.addHandler(fh)
    return log

# Save checkpoint
def save_checkpoint(state, filename="my_checkpoint.pth.tar"):
    """Save model checkpoint

    Parameters
    ----------
    state : dict
        dict contains model's state_dict, may contain
        other keys such as optimizer, epoch, etc.
    filename : str, optional
        path to save checkpoint file, by default "my_checkpoint.pth.tar"
    """    
    # print("=> Saving checkpoint")
    if not os.path.exists(os.path.dirname(filename)):
        print("Creating checkpoint directory {}".format(os.path.dirname(filename)))
        os.makedirs(os.path.dirname(filename))
    torch.save(state, filename)

# Load checkpoint
def load_checkpoint(checkpoint, model, optimizer=None):
    """Load model from checkpoint file

    Parameters
    ----------
    checkpoint : torch.load
        checkpoint file to be loaded
    model : torch.nn.Module
        model to be loaded
    optimizer : torch.optim
        optimizer to be loaded
    """    
    # print("=> Loading checkpoint from {}".format(checkpoint))
    model.load_state_dict(checkpoint["state_dict"], strict=False)
    if optimizer:
        optimizer.load_state_dict(checkpoint["optimizer"])

def NME(y_true, y_pred, device=torch.device("cpu")):
    """Calculate Normalized Mean Error

    Parameters
    ----------
    y_true : torch.Tensor
        ground truth landmarks
    y_pred : torch.Tensor
        predicted landmarks
    device : torch.device
        device on which model is loaded

    Returns
    -------
    nme : torch.Tensor
        Normalized Mean Error
    """    
    # print(y_true.shape, y_pred.shape)
    B = y_true.shape[0]
    y_true = y_true.to(device).reshape(B, -1, 2)
    y_pred = y_pred.to(device).reshape(B, -1, 2)
    if y_true.shape[1] == 44:
        interocular_distance = torch.linalg.norm(y_true[:, 20, :] - y_true[:, 29, :], dim=1)
    elif y_true.shape[1] == 5:
        interocular_distance = torch.linalg.norm(y_true[:, 0, :] - y_true[:, 1, :], dim=-1)
    else:
        raise ValueError("Number of landmarks is not 44 or 5")
    # print(interocular_distance.shape, y_true[:, 1, :].shape, interocular_distance)
    # nme = torch.linalg.norm(y_true - y_pred, dim=-1)
    # print(nme.mean().shape, interocular_distance)
    # nme = torch.sum(nme, dim=1) / (interocular_distance*y_pred.shape[1])
    # print(nme.shape)
    nme = torch.mean(torch.linalg.norm(y_true - y_pred, dim=-1) ) / interocular_distance
    # nme = torch.mean(torch.linalg.norm(y_true - y_pred, dim=-1) , dim=1)
    # print(nme)
    return torch.mean(nme)


def train(loop, model, optimizer, criterion, writer, step, device):
    """Train the model

    Parameters
    ----------
    loop : tqdm
        tqdm loop for training
    model : torch.nn.Module
        model to be trained
    optimizer : torch.optim
        optimizer to be used for training
    criterion : torch.nn
        loss function to be used for training
    writer : torch.utils.tensorboard.SummaryWriter
        tensorboard writer to be used for logging
    step : int
        current step of training
    device : torch.device
        device on which model is loaded

    Returns
    -------
    step : int
        updated step of training
    """    
    model.train()
    nme_total = 0
    for batch_idx, (data, targets) in enumerate(loop):
        # Get data to cuda if possible
        data = data.to(device)
        targets = targets.to(device)

        # forward
        predictions = model(data)
        # print(predictions[0], targets[0])
        loss = criterion(predictions, targets)

        # backward
        optimizer.zero_grad()
        loss.backward()

        # gradient descent or adam step
        optimizer.step()
        nme = NME(targets, predictions, device)
        nme_total += nme

        # update progress bar
        loop.set_postfix({"Loss": f"{loss.item():.4f}", "NME": f"{nme_total.item()/(batch_idx+1):.4f}"})
        # tensorboard
        writer.add_scalar("Training loss", loss, global_step=step)
        writer.add_scalar("Training NME", nme_total.item()/(batch_idx+1), global_step=step)
        step += 1

    return step, nme_total / len(loop)


def test(loop, model, criterion, writer, step, device):
    """test the model

    Parameters
    ----------
    loop : tqdm
        tqdm loop for testing
    model : torch.nn.Module
        model to be tested
    criterion : torch.nn
        loss function to be used for testing
    writer : torch.utils.tensorboard.SummaryWriter
        tensorboard writer to be used for logging
    step : int
        current step of testing
    device : torch.device
        device on which model is loaded

    Returns
    -------
    step : int
        updated step of testing
    """    
    model.eval()
    nme_total = 0
    with torch.no_grad():
        for batch_idx, (data, targets) in enumerate(loop):
            # Get data to cuda if possible
            data = data.to(device)
            targets = targets.to(device)

            # forward
            predictions = model(data)
            loss = criterion(predictions, targets)
            nme = NME(targets, predictions, device)
            nme_total += nme
            # print(nme_total)

            # update progress bar
            loop.set_postfix({"Loss": f"{loss.item():.4f}", "NME": f"{nme_total.item()/(batch_idx+1):.4f}"})

            # tensorboard
            writer.add_scalar("Testing loss", loss, global_step=step)
            writer.add_scalar("Testing NME", nme_total.item()/(batch_idx+1), global_step=step)
            step += 1
    mean_nme = nme_total / len(loop)
    return step, mean_nme

def Recover(image, landmarks, angle=0, size=(256, 256)):
    """Recover the image and landmarks to original angle and unnormalized form

    Parameters
    ----------
    image : torch.Tensor or np.ndarray
        image to be recovered, shape : (C, H, W) for torch.Tensor and (H, W, C) for np.ndarray
    landmarks : torch.Tensor or np.ndarray
        landmarks to be recovered, shape : (N, 2) or (1, N*2)
    angle : int, optional
        the angle of rotation, by default 0
    size : tuple, optional
        the size of the image to recover to, by default (256, 256)

    Returns
    -------
    new_img : np.ndarray
        recovered image
    new_landmarks : np.ndarray
        recovered landmarks
    """    
    # squeeze the image if it has a batch dimension
    if len(image.shape) == 4 and image.shape[0] == 1:
        image = image.squeeze(0)
    # unnormalize the image and landmarks if they are torch.Tensor outputs from the model
    # convert them to numpy.ndarray
    if isinstance(image, torch.Tensor):
        image, landmarks = torchlm.LandmarksUnNormalize()(image, landmarks)
        image, landmarks = image.numpy().astype(np.uint8), landmarks.numpy()
    # transpose the image if it is in the form of (C, H, W)
    if image.shape[0] == 3 or image.shape[0] == 1:
        image = image.transpose(1, 2, 0)
    # reshape the landmarks if they are in the form of (1, N*2)
    if landmarks.shape[-1] != 2:
        landmarks = landmarks.reshape(-1, 2)

    # do nothing if the angle is 0
    if angle == 0:
        new_img, new_landmarks = image, landmarks
    else:
        # get the center of the image
        w, h = image.shape[1], image.shape[0]
        cx, cy = w // 2, h // 2

        # grab the rotation matrix (applying the negative of the
        # angle to rotate clockwise), then grab the sine and cosine
        # (i.e., the rotation components of the matrix)
        M = cv2.getRotationMatrix2D((cx, cy), -angle, 1.0)

        # perform the actual rotation and return the image
        new_img = cv2.warpAffine(image.copy(), M, (w, h))

        new_landmarks = np.hstack((landmarks, np.ones((landmarks.shape[0], 1), dtype=type(landmarks[0][0]))))
        new_landmarks = np.dot(M, new_landmarks.T).T

    # resize the image and landmarks if the size is not the same as the target size
    if new_img.shape[0] != size[0] or new_img.shape[1] != size[1]:
        new_img, new_landmarks = torchlm.LandmarksResize(size)(new_img, new_landmarks)

    return new_img.astype(np.uint8), new_landmarks.astype(np.float32)

def Inferencer(model, image, recover=False, landmarks=None, angle=0, device="cpu", raw_image=None, raw_landmark=None, plot=False): 
    """Inference the model on the given image

    Parameters
    ----------
    model : torch.nn.Module
        model to be used for inference
    image : torch.Tensor
        image to be used for inference
    recover : bool, optional
        whether to recover the image and landmarks, by default False
    landmarks : _type_, optional
        ground truth landmarks, by default None
    angle : int, optional
        the angle of rotation for recovering, by default 0
    device : str, optional
        device on which model is loaded, by default "cpu"
    raw_image : np.ndarray, optional
        raw image to be plotted, by default None
    raw_landmark : np.ndarray, optional
        raw landmarks to be plotted, by default None
    plot : bool, optional
        whether to plot the image and landmarks, by default False

    Returns
    -------
    image : torch.Tensor
        image, recovered if recover is True
    pred : np.ndarray
        predicted landmarks, recovered if recover is True
    """    
    fake_image = torch.zeros_like(image)
    image = image.to(device)
    output = model(image).cpu().detach()
    image = image.cpu().detach()
    if recover:
        image, pred = Recover(image, output, angle)
        if landmarks is not None:
            fake_image, landmarks = Recover(fake_image, landmarks, angle)
    else:
        image, pred = Recover(image, output, size=image.shape[2:])
    if raw_image is None:
        raw_image = image
    if plot:
        plt.imshow(raw_image)
        plt.plot(pred[:, 0], pred[:, 1], 'bx')
        if landmarks is not None:
            if raw_landmark is None:
                raw_landmark = landmarks
            plt.plot(raw_landmark[:, 0], raw_landmark[:, 1], 'rx')
        plt.show()
    return image, pred

def save_as_csv(points, location = './results'):
    """
    Save the points out as a .csv file
    :param points: numpy array of shape (no_test_images, no_points, 2) to be saved
    :param location: Directory to save results.csv in. Default to current working directory
    """
    assert points.shape[0]==554, 'wrong number of image points, should be 554 test images'
    assert np.prod(points.shape[1:])==44*2, 'wrong number of points provided. There should be 34 points with 2 values (x,y) per point'
    np.savetxt(location + '/results.csv', np.reshape(points, (points.shape[0], -1)), delimiter=',')



if __name__ == "__main__":
    from resnet import *
    from datasets import FaceDataset
    from torch.utils.data import DataLoader
    # DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
    DEVICE = "cpu"
    NUM_OUTPUTS = 44*2  # 44 points, 2 coordinates
    TEST_PATH = "data/training_images_full_test.npz"
    model = resnet50(pretrained=True, num_classes=NUM_OUTPUTS).to(DEVICE)
    test_dataset = FaceDataset(path=TEST_PATH, partial=False, augment=False)
    criterion = torch.nn.MSELoss()
    for b in [1, 2, 4, 8, 16, 32, 64]:
        test_loader = DataLoader(dataset=test_dataset, batch_size=b, shuffle=True)
        test_loop = tqdm(test_loader, total=len(test_loader), leave=False)
        test_loop.set_description("Testing")
        # timestamp = datetime.datetime.now().strftime("%Y_%m_%d-%H_%M_%S")
        # writer = SummaryWriter(f"runs/{timestamp}/")
        for batch_idx, (image, landmarks) in enumerate(test_loop):
            image, landmarks = image.to(DEVICE), landmarks.to(DEVICE)
            print(image.shape)
            pred = model(image)
            print(pred.shape)
            # writer.add_graph(model, image)
            break
        break
        step = 0
        step, nme = test(test_loop, model, criterion, writer, step, DEVICE)
        print(b, nme)
    # test_loader = DataLoader(dataset=test_dataset, batch_size=4, shuffle=True)
    # test_loop = tqdm(test_loader, total=len(test_loader), leave=False)
    # test_loop.set_description("Testing")
    # # print(len(test_loop))
    # timestamp = datetime.datetime.now().strftime("%Y_%m_%d-%H_%M_%S")
    # writer = SummaryWriter(f"runs/{timestamp}/")
    # step = 0
    # step, nme = test(test_loop, model, criterion, writer, step, DEVICE)
    # print(nme)



# if __name__ == "__main__":
#     B = [8, 16, 32]
#     for b in B:
#         pred = torch.randn((b, 10))
#         target = torch.randn((b, 10))
#         nme = NME(pred, target)
#         print(b, nme)