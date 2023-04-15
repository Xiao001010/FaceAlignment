import math
import torch
import torch.nn as nn


class WingLoss(nn.Module):
    """Wing Loss for Facial Landmark Detection
    Wing Loss. paper ref: 'Wing Loss for Robust Facial Landmark Localisation
    with Convolutional Neural Networks' Feng et al. CVPR'2018.

    Parameters
    ----------
    omega : float, optional
        wing loss hyperparameter, Also referred to as width, by default 10.0
    epsilon : float, optional
        wing loss hyperparameter, Also referred to as curvature, by default 2.0
    """
    def __init__(self, omega=10.0, epsilon=2.0):

        super(WingLoss, self).__init__()

        self.omega = omega
        self.epsilon = epsilon
        # constant C in the paper that is used to 
        # smooth the piecewise-defined linear and non-linear parts of the loss
        self.C = self.omega * (1 - math.log(1 + self.omega / self.epsilon))

    def forward(self, y_pred, y_true):
        """Forward pass of the wing loss

        Notes
        -----
        batch_size: B
        num_landmarks: K
        dimension of landmarks: D (2 for 2D landmarks, 3 for 3D landmarks)

        Parameters
        ----------
        y_pred : torch.Tensor[B, K, D]
            predicted landmarks
        y_true : torch.Tensor[B, K, D]
            ground truth landmarks

        Returns
        -------
        loss : torch.Tensor[1]
            wing loss
        """
        # reshape the tensors to [B, K, D]
        if len(y_pred.shape) == 2:
            B = y_pred.shape[0]
            y_pred = y_pred.reshape(B, -1, 2)
            y_true = y_true.reshape(B, -1, 2)
        # calculate the absolute difference between the predicted and ground truth landmarks
        diff = torch.abs(y_pred - y_true)
        # calculate the loss for the linear part of the wing loss
        linear_losses = self.omega * torch.log(1 + diff / self.epsilon)
        # calculate the loss for the non-linear part of the wing loss
        non_linear_losses = diff - self.C
        # combine the linear and non-linear parts of the loss
        losses = torch.where(diff < self.omega, linear_losses, non_linear_losses)
        # return the mean of the loss
        return torch.mean(torch.sum(losses, dim=[1, 2]), dim=0)


class SoftWingLoss(nn.Module):
    """Soft Wing Loss 'Structure-Coherent Deep Feature Learning for Robust Face
    Alignment' Lin et al. TIP'2021.

    loss =
        1. |x|                           , if |x| < omega1
        2. omega2*ln(1+|x|/epsilon) + B, if |x| >= omega1
    
    Args:
        omega1 : float, optional
            The first threshold, by default 2.0
        omega2 : float, optional
            The second threshold, by default 20.0
        epsilon : float, optional
            Also referred to as curvature, by default 0.5
    """

    def __init__(self, omega1=2.0, omega2=20.0, epsilon=0.5):
        super(SoftWingLoss, self).__init__()
        self.omega1 = omega1
        self.omega2 = omega2
        self.epsilon = epsilon
        
        # constant that smoothly links the piecewise-defined linear
        # and nonlinear parts
        self.B = self.omega1 - self.omega2 * math.log(1.0 + self.omega1 / self.epsilon)

    def forward(self, y_pred, y_true):
        """Forward pass of the soft wing loss

        Notes
        -----
        batch_size: B
        num_landmarks: K
        dimension of landmarks: D (2 for 2D landmarks, 3 for 3D landmarks)


        Parameters
        ----------
        y_pred : torch.Tensor[B, K, D]
            predicted landmarks
        y_true : torch.Tensor[B, K, D]
            ground truth landmarks

        Returns
        -------
        loss : torch.Tensor[1]
            soft wing loss
        """        
        # reshape the tensors to [B, K, D]
        if len(y_pred.shape) == 3:
            B = y_pred.shape[0]
            y_pred = y_pred.reshape(B, -1, 2)
            y_true = y_true.reshape(B, -1, 2)
        diff = torch.abs(y_pred - y_true)
        losses = torch.where(diff < self.omega1, diff, self.omega2 * torch.log(1.0 + diff / self.epsilon) + self.B)
        return torch.mean(torch.sum(losses, dim=[1, 2]), dim=0)


# Adaptiive Wing Loss needs to be used with heatmaps, not landmarks
# So, it is not used in this project
class AdaptiveWingLoss(nn.Module):
    """Adaptive wing loss. paper ref: 'Adaptive Wing Loss for Robust Face
    Alignment via Heatmap Regression' Wang et al. ICCV'2019.

    Args:
        alpha : float, optional
            adaptive wing loss hyperparameter, by default 2.0
        omega : float, optional
            adaptive wing loss hyperparameter, by default 14.0
        epsilon : float, optional
            adaptive wing loss hyperparameter, by default 1.0
    """

    def __init__(self, alpha=2.0, omega=14.0, epsilon=1.0, theta=0.5):
        super(AdaptiveWingLoss, self).__init__()
        self.alpha = alpha
        self.omega = omega
        self.epsilon = epsilon
        self.theta = theta

    def forward(self, y_pred, y_true):
        """Forward pass of the adaptive wing loss

        Notes
        -----
        batch_size: B
        num_landmarks: K

        Parameters
        ----------
        y_pred : torch.Tensor[B, K, H, W]
            predicted heatmaps
        y_true : torch.Tensor[B, K, H, W]
            ground truth heatmaps

        Returns
        -------
        loss : torch.Tensor[1]
            adaptive wing loss
        """        
        H, W = y_pred.shape[-2:]
        # calculate the absolute difference between the predicted and ground truth landmarks
        diff = torch.abs(y_pred - y_true)
        
        A = self.omega * (
            1 / (1 + torch.pow(self.theta / self.epsilon, self.alpha - y_true))
        ) * (self.alpha - y_true) * (torch.pow(
            self.theta / self.epsilon,
            self.alpha - y_true - 1)) * (1 / self.epsilon)
        C = self.theta * A - self.omega * torch.log(
            1 + torch.pow(self.theta / self.epsilon, self.alpha - y_true))
        losses = torch.where(
            diff < self.theta,
            self.omega * torch.log(1 + torch.pow(diff / self.epsilon, self.alpha)),
            A * diff - C)
        return torch.mean(losses)
    

if __name__ == "__main__":
    criteria = WingLoss()
    B = [8, 16, 32]
    for b in B:
        pred = torch.ones((b, 10))
        target = torch.zeros((b, 10))
        loss = criteria(pred, target)
        print(b, loss)