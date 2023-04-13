import cv2
import math
import torch
import numpy as np

import torchlm
# from typing import Tuple, Union, List, Optional, Callable, Any, Dict
# from torch import Tensor



# def to_tensor(x: Union[np.ndarray, Tensor]) -> Tensor:
#     """without post process, such as normalize"""
#     assert isinstance(x, (np.ndarray, Tensor))

#     if isinstance(x, np.ndarray):
#         return torch.from_numpy(x)
#     return x


# def to_numpy(x: Union[np.ndarray, Tensor]) -> np.ndarray:
#     """without post process, such as transpose"""
#     assert isinstance(x, (np.ndarray, Tensor))

#     if isinstance(x, np.ndarray):
#         return x
#     try:
#         return x.cpu().numpy()
#     except:
#         return x.detach().cpu().numpy()

# # # base element_type
# # Base_Element_Type = Union[np.ndarray, Tensor]
# # Image_InOutput_Type = Base_Element_Type  # image
# # Landmarks_InOutput_Type = Base_Element_Type  # landmarks

# # class AutoDtypeEnum:
# #     # autodtype modes
# #     Array_In: int = 0
# #     Array_InOut: int = 1
# #     Tensor_In: int = 2
# #     Tensor_InOut: int = 3

# # AutoDtypeLoggingMode: bool = False

# # def set_autodtype_logging(logging: bool = False):
# #     global AutoDtypeLoggingMode
# #     AutoDtypeLoggingMode = logging

# # def _autodtype_api_logging(self: Any, mode: int):
# #     global AutoDtypeLoggingMode
# #     if AutoDtypeLoggingMode:
# #         mode_info_map: Dict[int, str] = {
# #             AutoDtypeEnum.Array_In: "AutoDtypeEnum.Array_In",
# #             AutoDtypeEnum.Array_InOut: "AutoDtypeEnum.Array_InOut",
# #             AutoDtypeEnum.Tensor_In: "AutoDtypeEnum.Tensor_In",
# #             AutoDtypeEnum.Tensor_InOut: "AutoDtypeEnum.Tensor_InOut"
# #         }
# #         print(f"{self}() AutoDtype Info: {mode_info_map[mode]}")

# # def autodtype(mode: int) -> Callable:
# #     # A Pythonic style to auto convert input dtype and let the output dtype unchanged

# #     assert 0 <= mode <= 3

# #     def wrapper(
# #             callable_array_or_tensor_func: Callable
# #     ) -> Callable:

# #         def apply(
# #                 self,
# #                 img: Image_InOutput_Type,
# #                 landmarks: Landmarks_InOutput_Type,
# #                 **kwargs
# #         ) -> Tuple[Image_InOutput_Type, Landmarks_InOutput_Type]:
# #             # Type checks
# #             assert all(
# #                 [isinstance(_, (np.ndarray, Tensor))
# #                  for _ in (img, landmarks)]
# #             ), "Error dtype, must be np.ndarray or Tensor!"
# #             # force array before transform and then wrap back.
# #             if mode == AutoDtypeEnum.Array_InOut:
# #                 _autodtype_api_logging(self, mode)

# #                 if any((
# #                         isinstance(img, Tensor),
# #                         isinstance(landmarks, Tensor)
# #                 )):
# #                     img = to_numpy(img)
# #                     landmarks = to_numpy(landmarks)
# #                     img, landmarks = callable_array_or_tensor_func(
# #                         self,
# #                         img,
# #                         landmarks,
# #                         **kwargs
# #                     )
# #                     img = to_tensor(img)
# #                     landmarks = to_tensor(landmarks)
# #                 else:
# #                     img, landmarks = callable_array_or_tensor_func(
# #                         self,
# #                         img,
# #                         landmarks,
# #                         **kwargs
# #                     )
# #             # force array before transform and don't wrap back.
# #             elif mode == AutoDtypeEnum.Array_In:
# #                 _autodtype_api_logging(self, mode)

# #                 if any((
# #                         isinstance(img, Tensor),
# #                         isinstance(landmarks, Tensor)
# #                 )):
# #                     img = to_numpy(img)
# #                     landmarks = to_numpy(landmarks)
# #                     img, landmarks = callable_array_or_tensor_func(
# #                         self,
# #                         img,
# #                         landmarks,
# #                         **kwargs
# #                     )
# #                 else:
# #                     img, landmarks = callable_array_or_tensor_func(
# #                         self,
# #                         img,
# #                         landmarks,
# #                         **kwargs
# #                     )
# #             # force tensor before transform and then wrap back.
# #             elif mode == AutoDtypeEnum.Tensor_InOut:
# #                 _autodtype_api_logging(self, mode)

# #                 if any((
# #                         isinstance(img, np.ndarray),
# #                         isinstance(landmarks, np.ndarray)
# #                 )):
# #                     img = to_tensor(img)
# #                     landmarks = to_tensor(landmarks)
# #                     img, landmarks = callable_array_or_tensor_func(
# #                         self,
# #                         img,
# #                         landmarks,
# #                         **kwargs
# #                     )
# #                     img = to_numpy(img)
# #                     landmarks = to_numpy(landmarks)
# #                 else:
# #                     img, landmarks = callable_array_or_tensor_func(
# #                         self,
# #                         img,
# #                         landmarks,
# #                         **kwargs
# #                     )
# #             # force tensor before transform and don't wrap back.
# #             elif mode == AutoDtypeEnum.Tensor_In:
# #                 _autodtype_api_logging(self, mode)

# #                 if any((
# #                         isinstance(img, np.ndarray),
# #                         isinstance(landmarks, np.ndarray)
# #                 )):
# #                     img = to_tensor(img)
# #                     landmarks = to_tensor(landmarks)
# #                     img, landmarks = callable_array_or_tensor_func(
# #                         self,
# #                         img,
# #                         landmarks,
# #                         **kwargs
# #                     )
# #                 else:
# #                     img, landmarks = callable_array_or_tensor_func(
# #                         self,
# #                         img,
# #                         landmarks,
# #                         **kwargs
# #                     )
# #             else:
# #                 _autodtype_api_logging(self, mode)
# #                 img, landmarks = callable_array_or_tensor_func(
# #                     self,
# #                     img,
# #                     landmarks,
# #                     **kwargs
# #                 )

# #             return img, landmarks

# #         return apply

# #     return wrapper


# # def _transforms_api_assert(self: torchlm.LandmarksTransform, cond: bool, info: str = None):
# #     if cond:
# #         self.flag = False  # flag is a reference of some specific flag
# #         if info is None:
# #             info = f"{self}() missing landmarks"
# #         raise ValueError(info)




# class CustomerAlign(torchlm.LandmarksAlign):
#     """Get aligned image and landmarks"""

#     def __init__(
#             self,
#             eyes_index: Union[Tuple[int, int], List[int]] = None
#     ):
#         """
#         :param eyes_index: 2 indexes in landmarks, indicates left and right eye center.
#         """
#         super(torchlm.LandmarksAlign, self).__init__()
#         if eyes_index is None or len(eyes_index) != 2:
#             raise ValueError("2 indexes in landmarks, "
#                              "which indicates left and right eye center.")

#         self._eyes_index = eyes_index

#     def __call__(
#             self,
#             img: np.ndarray,
#             landmarks: np.ndarray
#     ) -> Tuple[np.ndarray, np.ndarray]:
#         left_eye = landmarks[self._eyes_index[0]]
#         right_eye = landmarks[self._eyes_index[1]]
#         dx = (right_eye[0] - left_eye[0])
#         dy = (right_eye[1] - left_eye[1])
#         angle = math.atan2(dy, dx) * 180 / math.pi  # calc angle

#         try:
#             img = img.cpu().numpy()
#         except:
#             img = img.detach().cpu().numpy()
#         print(img.shape)

#         w, h = img.shape[1], img.shape[0]
#         cx, cy = w // 2, h // 2

#         # grab the rotation matrix (applying the negative of the
#         # angle to rotate clockwise), then grab the sine and cosine
#         # (i.e., the rotation components of the matrix)
#         M = cv2.getRotationMatrix2D((cx, cy), angle, 1.0)

#         # perform the actual rotation and return the image
#         new_img = cv2.warpAffine(img.copy(), M, (w, h))

#         new_landmarks = np.hstack((landmarks, np.ones((landmarks.shape[0], 1), dtype=type(landmarks[0][0]))))
#         new_landmarks = np.dot(M, new_landmarks.T).T

#         self.flag = True

#         return new_img.astype(np.uint8), new_landmarks.astype(np.float32), angle.astype(np.float32)
    

# class LandmarksRecover(torchlm.LandmarksAlign):
#     """Get aligned image and landmarks"""

#     def __init__(
#             self,
#             angle: float
#     ):
#         """
#         :param angle: angle to recover
#         """
#         super(LandmarksRecover, self).__init__()
#         self._angle = -angle

#     def __call__(
#             self,
#             img: np.ndarray,
#             landmarks: np.ndarray
#     ) -> Tuple[np.ndarray, np.ndarray]:

#         num_landmarks = len(landmarks)

#         w, h = img.shape[1], img.shape[0]
#         cx, cy = w // 2, h // 2

#         # grab the rotation matrix (applying the negative of the
#         # angle to rotate clockwise), then grab the sine and cosine
#         # (i.e., the rotation components of the matrix)
#         M = cv2.getRotationMatrix2D((cx, cy), self._angle, 1.0)

#         # perform the actual rotation and return the image
#         new_img = cv2.warpAffine(img.copy(), M, (w, h))

#         new_landmarks = np.hstack((landmarks, np.ones((landmarks.shape[0], 1), dtype=type(landmarks[0][0]))))
#         new_landmarks = np.dot(M, new_landmarks.T).T

#         self.flag = True

#         return new_img.astype(np.uint8), new_landmarks.astype(np.float32)









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