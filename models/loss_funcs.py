import torch
import torch.nn as nn
import torch.nn.functional as F

class DiceLoss(nn.Module):
    def __init__(self, smooth=1):
        super().__init__()
        self.smooth = smooth
    
    def forward(self, pred, target):
        pred, target = pred.float(), target.float()
        intersection = torch.sum(pred * target, dim=(2, 3)) # (N, C, H, W)
        union = torch.sum(pred, dim=(2, 3)) + torch.sum(target, dim=(2, 3))
        dice = (2 * (intersection + self.smooth)) / (union + self.smooth)
        return 1 - dice

class BoundaryLoss(nn.Module):
    """Boundary Loss proposed in:
    Alexey Bokhovkin et al., Boundary Loss for Remote Sensing Imagery Semantic Segmentation
    https://arxiv.org/abs/1905.07852
    Inspired from https://github.com/yiskw713/boundary_loss_for_remote_sensing
    """

    def __init__(self, theta0=3, theta=5, smooth=1e-7):
        super().__init__()
        self.theta0 = theta0
        self.theta = theta
        self.smooth = smooth

    def forward(self, pred, target):
        # boundary map
        gt_b = F.max_pool2d(
            1 - target, kernel_size=self.theta0, stride=1, padding=(self.theta0 - 1) // 2)
        gt_b -= 1 - target

        pred_b = F.max_pool2d(
            1 - pred, kernel_size=self.theta0, stride=1, padding=(self.theta0 - 1) // 2)
        pred_b -= 1 - pred

        # extended boundary map
        gt_b_ext = F.max_pool2d(
            gt_b, kernel_size=self.theta, stride=1, padding=(self.theta - 1) // 2)

        pred_b_ext = F.max_pool2d(
            pred_b, kernel_size=self.theta, stride=1, padding=(self.theta - 1) // 2)
        
        # Precision, Recall
        P = torch.sum(pred_b * gt_b_ext, dim=(2, 3)) / (torch.sum(pred_b, dim=(2, 3)) + self.smooth)
        R = torch.sum(pred_b_ext * gt_b, dim=(2, 3)) / (torch.sum(gt_b, dim=(2, 3)) + self.smooth)

        # Boundary F1 Score
        BF1 = (2 * P * R) / (P + R + self.smooth)

        # summing BF1 Score for each class and average over mini-batch
        return torch.mean(1 - BF1)


if __name__ == '__main__':
    dummy_pred = torch.randint(low=0, high=2, size=(1, 3, 192, 192))
    dummy_target = torch.randint(low=0, high=2, size=(1, 3, 192, 192))

    diceloss = DiceLoss()
    boundaryloss = BoundaryLoss()

    print(f'Dice Loss: {diceloss(dummy_pred, dummy_target)}')
    print(f'Boundary Loss: {boundaryloss(dummy_pred, dummy_target)}')