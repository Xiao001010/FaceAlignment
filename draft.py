import cv2
import math
import torch
import numpy as np

import torchlm

def Align(image, landmarks, eyes_index):
    if eyes_index is None or len(eyes_index) != 2:
        raise ValueError("2 indexes in landmarks, "
                            "which indicates left and right eye center.")

    left_eye = landmarks[eyes_index[0]]
    right_eye = landmarks[eyes_index[1]]
    dx = (right_eye[0] - left_eye[0])
    dy = (right_eye[1] - left_eye[1])
    angle = math.atan2(dy, dx) * 180 / math.pi  # calc angle

    w, h = image.shape[1], image.shape[0]
    cx, cy = w // 2, h // 2

    # grab the rotation matrix (applying the negative of the
    # angle to rotate clockwise), then grab the sine and cosine
    # (i.e., the rotation components of the matrix)
    M = cv2.getRotationMatrix2D((cx, cy), angle, 1.0)

    # perform the actual rotation and return the image
    new_img = cv2.warpAffine(image.copy(), M, (w, h))

    new_landmarks = np.hstack((landmarks, np.ones((landmarks.shape[0], 1), dtype=type(landmarks[0][0]))))
    new_landmarks = np.dot(M, new_landmarks.T).T

    return new_img.astype(np.uint8), new_landmarks.astype(np.float32), angle


def Recover(image, landmarks, angle):
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

    return new_img.astype(np.uint8), new_landmarks.astype(np.float32)

from matplotlib import pyplot as plt
from torch.utils.data import DataLoader
from datasets import FaceDataset

dataset = FaceDataset("data/training_images_full.npz", partial=True, augment=False)
loader = DataLoader(dataset, batch_size=5, shuffle=False)

for idx, (img_batch, pts_batch) in enumerate(loader):
    img_batch, pts_batch  = torchlm.LandmarksUnNormalize()(img_batch, pts_batch)
    print(img_batch.shape, pts_batch.shape)
    pts = pts_batch[1].reshape(-1, 2)
    img = img_batch[1].numpy()
    img = img.astype(np.uint8)
    print(pts.shape, img.shape)
    img = np.transpose(img, (1, 2, 0))
    # print(pts)
    plt.imshow(img)
    plt.plot(pts[:, 0], pts[:, 1], 'bx')
    plt.show()
    break

print("Align")
trans = torchlm.LandmarksCompose([torchlm.LandmarksResize((224, 224)), 
                                  torchlm.LandmarksNormalize(),
                                  torchlm.LandmarksToTensor(),
                                  ])
angles = []
for idx, (img, pts) in enumerate(zip(img_batch, pts_batch)):
    # print(idx, img.shape, pts.shape)
    # print(idx, img_ro.shape, pts_ro.shape)
    # img_ro = np.transpose(img_ro.clone().cpu().numpy(), (2, 0, 1))
    img_ro = np.transpose(img.clone().cpu().numpy(), (1, 2, 0))
    # print(idx, img_ro.shape, pts.shape)
    img_ro, pts_ro = trans(img_ro, pts.clone().cpu().numpy().reshape(-1, 2))
    img_ro = np.transpose(img_ro.clone().cpu().numpy(), (1, 2, 0))
    # print(idx, img_ro.shape, pts_ro.shape)
    if idx==1:
        print("1", pts_ro)
    # img_ro, pts_ro = trans(img_ro, pts_ro)
    pts_ro = pts.clone().cpu().numpy().reshape(-1, 2)
    img_ro, pts_ro, angle = Align(img_ro, pts_ro, [0, 1])
    # print(idx, img_ro)

    # print(idx, img_ro)
    img_batch = torch.zeros((5, 3, 224, 224))
    img_batch[idx] = torch.from_numpy(np.transpose(img_ro, (2, 0, 1)))
    pts_batch[idx] = torch.from_numpy(pts_ro.reshape(-1))
    # img_batch[idx] = img_ro
    # pts_batch[idx] = pts_ro.reshape(-1)
    angles.append(angle)
    print(pts_ro.shape, img_ro.shape)
    # print(pts_ro)

print(img_batch.shape, pts_batch.shape)
img_batch, pts_batch  = torchlm.LandmarksUnNormalize()(img_batch, pts_batch)
pts = pts_batch[1].reshape(-1, 2)
img = img_batch[1].numpy()
img = img.astype(np.uint8)
# print("img", img)
print(pts.shape, img.shape)
img = np.transpose(img, (1, 2, 0))
print(pts)
plt.imshow(img)
plt.plot(pts[:, 0], pts[:, 1], 'bx')
plt.show()


print("Recover")
for idx, (img, pts, angle) in enumerate(zip(img_batch, pts_batch, angles)):
    img_ro = np.transpose(img.clone().cpu().numpy(), (1, 2, 0))
    pts_ro = pts.clone().cpu().numpy().reshape(-1, 2)
    angle = angles[idx]
    img_ro, pts_ro = Recover(img_ro, pts_ro, angle)
    img_batch[idx] = torch.from_numpy(np.transpose(img_ro, (2, 0, 1)))
    pts_batch[idx] = torch.from_numpy(pts_ro.reshape(-1))
    # print(pts_ro.shape, img_ro.shape)
    # print(pts_ro)

print(img_batch.shape, pts_batch.shape)
pts = pts_batch[1].reshape(-1, 2)
img = img_batch[1].numpy()
img = img.astype(np.uint8)
print(pts.shape, img.shape)
img = np.transpose(img, (1, 2, 0))
print("1", pts)
plt.imshow(img)
plt.plot(pts[:, 0], pts[:, 1], 'bx')
plt.show()