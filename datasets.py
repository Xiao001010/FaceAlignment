import math
import cv2
import numpy as np
import torch
from torch.utils.data import Dataset
# from torchvision import transforms
import torchlm


def split_data(path, split=0.8):
    """This function splits the data into train and test data. The split ratio is 80:20 by default.

    Parameters
    ----------
    path : str
        path to the data
    split : float, optional
        split ratio, by default 0.8
    """    
    # Load the data using np.load
    data = np.load(path, allow_pickle=True)

    # Extract the images
    images = data['images']
    # and the data points
    pts = data['points']

    # Split the data
    split_idx = int(split * len(images))
    train_images = images[:split_idx]
    train_pts = pts[:split_idx]
    test_images = images[split_idx:]
    test_pts = pts[split_idx:]

    # Save the data
    np.savez_compressed(path.replace(".npz", "_train.npz"), images=train_images, points=train_pts)
    np.savez_compressed(path.replace(".npz", "_test.npz"), images=test_images, points=test_pts)


def Light(image, landmarks):
    """This function adds light to the image. The light is added at a random position 
    and the radius of the light is also random. 
    The light is added by increasing the pixel values of the image.

    Notes
    -----
    Width : W
    Height : H
    Number of landmarks : K

    Parameters
    ----------
    image : np.ndarray[W, H, 3]
        image to which light is to be added
    landmarks : np.ndarray[K, 2]
        landmarks of the image

    Returns
    -------
    image : np.ndarray[W, H, 3]
        image with light added
    landmarks : np.ndarray[K, 2]
        landmarks of the image
    """    
    # get the image size
    x, y,_ = image.shape
    radius = np.random.randint(10, int(min(x, y)), 1)

    # get the center of the light
    pos_x = np.random.randint(0, (min(x, y) - radius), 1) 
    pos_y = np.random.randint(0, (min(x, y) - radius), 1)
    pos_x = int(pos_x[0])
    pos_y = int(pos_y[0])
    radius = int(radius[0])

    # light strength
    strength = 50
    for j in range(pos_y - radius, pos_y + radius):
        for i in range(pos_x-radius, pos_x+radius):

            # distance to the center of the light
            distance = math.pow((pos_x - i), 2) + math.pow((pos_y - j), 2)
            distance = np.sqrt(distance)
            if distance < radius:
                result = 1 - distance / radius
                result = result*strength
                image[i, j, 0] = min((image[i, j, 0] + result),255)
                image[i, j, 1] = min((image[i, j, 1] + result),255)
                image[i, j, 2] = min((image[i, j, 2] + result),255)
    image = image.astype(np.uint8)
    return image, landmarks


def Shadow(image, landmarks):
    """This function adds shadow to the image. The shadow is added at a random position 
    and the radius of the shadow is also random.
    The shadow is added by decreasing the pixel values of the image.

    Notes
    -----
    Width : W
    Height : H
    Number of landmarks : K

    Parameters
    ----------
    image : np.ndarray[W, H, 3]
        image to which shadow is to be added
    landmarks : np.ndarray[K, 2]
        landmarks of the image

    Returns
    -------
    image : np.ndarray[W, H, 3]
        image with shadow added
    landmarks : np.ndarray[K, 2]
        landmarks of the image
    """    
    # get the image size
    x, y,_ = image.shape
    radius = np.random.randint(10, int(min(x, y)), 1)

    # get the center of the light
    pos_x = np.random.randint(0, (min(x, y) - radius), 1)
    pos_y = np.random.randint(0, (min(x, y) - radius), 1)
    pos_x = int(pos_x[0])
    pos_y = int(pos_y[0])
    radius = int(radius[0])

    # light strength
    strength = 50
    for j in range(pos_y - radius, pos_y + radius):
        for i in range(pos_x-radius, pos_x+radius):

            # distance to the center of the light
            distance = math.pow((pos_x - i), 2) + math.pow((pos_y - j), 2)
            distance = np.sqrt(distance)
            if distance < radius:
                result = 1 - distance / radius
                result = result*strength
                image[i, j, 0] = max((image[i, j, 0] - result),0)
                image[i, j, 1] = max((image[i, j, 1] - result),0)
                image[i, j, 2] = max((image[i, j, 2] - result),0)
    image = image.astype(np.uint8)
    return image, landmarks

def Mask(image, landmarks):
    """This function adds a mask to the image. The mask is added at a random position
    and the size of the mask is also random.

    Notes
    ------
    Width : W
    Height : H
    Number of landmarks : K

    Parameters
    ----------
    image : np.ndarray[W, H, 3]
        image to which mask is to be added
    landmarks : np.ndarray[K, 2]
        landmarks of the image

    Returns
    -------
    image : np.ndarray[W, H, 3]
        image with mask added
    landmarks : np.ndarray[K, 2]
        landmarks of the image
    """    
    # get the image size
    x, y,_ = image.shape
    
    # get the size of the mask
    mask_size = np.random.randint(10, 40, 1)

    # get the left top corner of the mask
    pos_x = np.random.randint(10, (min(x, y) - 50), 1)
    pos_y = np.random.randint(10, (min(x, y) - 50), 1)
    pos_x = int(pos_x[0])
    pos_y = int(pos_y[0])
    mask_size = int(mask_size[0])
    image[pos_x:pos_x + mask_size, pos_y:pos_y + mask_size] = 0
    return image, landmarks



def Align(image, landmarks, eyes):
    left_eye = eyes[0]
    right_eye = eyes[1]
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



class FaceDataset(Dataset):
    """This class is used to load the dataset.

    Parameters
    ----------
    path : str
        path to the dataset
    partial : bool, optional
        if True, the dataset will be loaded in partial mode, 
        only 5 landmarks will be loaded, by default False
    augment : bool, optional
        if True, the dataset will be augmented, by default False
    """    
    def __init__(self, path, partial=False, augment=False):

        self.partial = partial
        self.augment = augment
        if self.augment:
            self.trans = torchlm.LandmarksCompose([
                                        torchlm.bind(Light, bind_type=torchlm.BindEnum.Callable_Array, prob=0.5),
                                        torchlm.bind(Shadow, bind_type=torchlm.BindEnum.Callable_Array, prob=0.5), 
                                        torchlm.bind(Mask, bind_type=torchlm.BindEnum.Callable_Array, prob=0.5),

                                        # flip the image horizontally will change the landmarks index sequence, 
                                        # so we need to change the landmarks index sequence
                                        # torchlm.LandmarksRandomHorizontalFlip(0.5),
                                        torchlm.LandmarksRandomTranslate(0.1, prob=0.5),
                                        torchlm.LandmarksRandomShear([-0.1, 0.1], prob=0.5),
                                        torchlm.LandmarksRandomRotate(10, prob=0.5),
                                        torchlm.LandmarksRandomScale([-0.1, 0.1], prob=0.5),

                                        # torchlm.LandmarksRandomMask(mask_ratio=0.021),
                                        torchlm.LandmarksRandomBlur(kernel_range=(3, 5), prob=0.5),
                                        # torchlm.LandmarksRandomBrightness((-5, 5), (0.8, 1.5)),
                                        torchlm.LandmarksResize((224, 224)), 
                                        torchlm.LandmarksNormalize(),
                                        torchlm.LandmarksToTensor(),
                                        ])

        else:
            self.trans = torchlm.LandmarksCompose([
                                        torchlm.LandmarksResize((224, 224)), 
                                        torchlm.LandmarksNormalize(),
                                        torchlm.LandmarksToTensor(),
                                        ])
            # self.trans = transforms.Compose([transforms.Resize((224, 224)),
            #                                  transforms.ToTensor(),
            #                                  transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))])

        # Load the data using np.load
        data = np.load(path, allow_pickle=True)
        # Extract the images
        self.images = data['images']
        # and the data points
        self.pts = data['points']

    def __getitem__(self, index):
        """This function is used to get the image and the landmarks of the image.

        Notes
        ------
        number of landmarks : K = 44 if partial is False else 5
        label shape : 44*2 if partial is False else 5*2

        Parameters
        ----------
        index : int
            index of the image

        Returns
        -------
        img : torch.Tensor[3, 224, 224]
            image
        label : torch.Tensor[88 if partial is False else 10]
            landmarks of the image
        """        
        img = self.images[index]
        label = self.pts[index]
        if self.partial and label.shape[-2] == 44:
            indices = (20, 29, 16, 32, 38)
            label = label[indices, :]
        # img = Image.fromarray(img.astype('uint8'), mode='RGB')
        if self.augment:
            img, label = self.trans(img, label)
        else:
            img, label = self.trans(img, label)
        return img, label.reshape(-1)

    def __len__(self):
        return len(self.images)
    

class CascadeStage2Dataset(Dataset):
    """This class is used to load the dataset.

    Parameters
    ----------
    path : str
        path to the dataset
    partial : bool, optional
        if True, the dataset will be loaded in partial mode, 
        only 5 landmarks will be loaded, by default False
    augment : bool, optional
        if True, the dataset will be augmented, by default False
    """    
    def __init__(self, path, model, augment=False, device=torch.device('cpu'), required_angle=False):
        self.model = model
        self.augment = augment
        self.device = device
        self.required_angle = required_angle
        if self.augment:
            self.trans = torchlm.LandmarksCompose([
                                        torchlm.bind(Light, bind_type=torchlm.BindEnum.Callable_Array, prob=0.5),
                                        torchlm.bind(Shadow, bind_type=torchlm.BindEnum.Callable_Array, prob=0.5), 
                                        torchlm.bind(Mask, bind_type=torchlm.BindEnum.Callable_Array, prob=0.5),

                                        torchlm.LandmarksRandomBlur(kernel_range=(3, 5), prob=0.5),
                                        torchlm.LandmarksResize((224, 224)), 
                                        torchlm.LandmarksNormalize(),
                                        torchlm.LandmarksToTensor(),
                                        ])

        else:
            self.trans = torchlm.LandmarksCompose([
                                        # I don't know why the landmarks will be changed if I resize the image
                                        # torchlm.LandmarksResize((224, 224)), 
                                        torchlm.LandmarksNormalize(),
                                        torchlm.LandmarksToTensor(),
                                        ])
            # self.trans = transforms.Compose([transforms.Resize((224, 224)),
            #                                  transforms.ToTensor(),
            #                                  transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))])

        # Load the data using np.load
        data = np.load(path, allow_pickle=True)
        # Extract the images
        self.images = data['images']
        # and the data points
        self.pts = data['points']

    def __getitem__(self, index):
        """This function is used to get the image and the landmarks of the image.

        Notes
        ------
        number of landmarks : K = 44 if partial is False else 5
        label shape : 44*2 if partial is False else 5*2

        Parameters
        ----------
        index : int
            index of the image

        Returns
        -------
        img : torch.Tensor[3, 224, 224]
            image
        label : torch.Tensor[88 if partial is False else 10]
            landmarks of the image
        """      
        with torch.no_grad():  
            pred = self.model(torch.tensor(self.images[index]).unsqueeze(0).permute(0, 3, 1, 2).float().to(self.device))
            eyes = pred.detach().cpu().numpy().reshape(-1, 2)[[0, 1]]
        img = self.images[index]
        label = self.pts[index]
        img, label, angle = Align(img, label, eyes)
        if self.augment:
            img, label = self.trans(img, label)
        else:
            img, label = self.trans(img, label)
        if self.required_angle:
            return img, label.reshape(-1), angle
        else:
            return img, label.reshape(-1)

    def __len__(self):
        return len(self.images)
    
if __name__ == "__main__":
    # split_data("data/training_images_full.npz", 0.8)

    dataset = FaceDataset("data/training_images_full.npz", partial=True, augment=False)
    print(len(dataset))
    # print(dataset[0][0].shape)
    # print(dataset[0][1].shape)
    # print(dataset[0][0])
    # print(dataset[0][1])
    # import cv2
    import matplotlib.pyplot as plt
    from torch.utils.data import DataLoader
    loader = DataLoader(dataset, batch_size=1, shuffle=True)
    for idx, (img, pts) in enumerate(loader):
        # img = np.transpose(img, (1, 2, 0))
        # img = img * 255
        img, pts  = torchlm.LandmarksUnNormalize()(img, pts)
        pts = pts.squeeze(0).reshape(-1, 2)
        img = img.squeeze(0).numpy()
        img = img.astype(np.uint8)
        img = np.transpose(img, (1, 2, 0))
        print(pts)
        # plt.imshow(img)
        plt.plot(pts[:, 0], pts[:, 1], 'bx')
        plt.show()
        if idx == 10:
            break