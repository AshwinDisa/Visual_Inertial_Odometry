import torch
from torch import nn
import pytorch_lightning as pl
from torch.utils.data import DataLoader

from models.helpers import BasicConvEncoder
from models.helpers import POLAUpdate
from models.helpers import DropPath
from models.helpers import to_2tuple

def quaternion_to_matrix(quaternions: torch.Tensor) -> torch.Tensor:
    """
    Convert rotations given as quaternions to rotation matrices.

    Args:
        quaternions: quaternions with real part first,
            as tensor of shape (..., 4).

    Returns:
        Rotation matrices as tensor of shape (..., 3, 3).
    """
    r, i, j, k = torch.unbind(quaternions, -1)
    # pyre-fixme[58]: `/` is not supported for operand types `float` and `Tensor`.
    two_s = 2.0 / (quaternions * quaternions).sum(-1)

    o = torch.stack(
        (
            1 - two_s * (j * j + k * k),
            two_s * (i * j - k * r),
            two_s * (i * k + j * r),
            two_s * (i * j + k * r),
            1 - two_s * (i * i + k * k),
            two_s * (j * k - i * r),
            two_s * (i * k - j * r),
            two_s * (j * k + i * r),
            1 - two_s * (i * i + j * j),
        ),
        -1,
    )
    return o.reshape(quaternions.shape[:-1] + (3, 3))


class GeodesicLoss(nn.Module):
    r"""Creates a criterion that measures the distance between rotation matrices, which is
    useful for pose estimation problems.
    The distance ranges from 0 to :math:`pi`.
    See: http://www.boris-belousov.net/2016/12/01/quat-dist/#using-rotation-matrices and:
    "Metrics for 3D Rotations: Comparison and Analysis" (https://link.springer.com/article/10.1007/s10851-009-0161-2).

    Both `input` and `target` consist of rotation matrices, i.e., they have to be Tensors
    of size :math:`(minibatch, 3, 3)`.

    The loss can be described as:

    .. math::
        \text{loss}(R_{S}, R_{T}) = \arccos\left(\frac{\text{tr} (R_{S} R_{T}^{T}) - 1}{2}\right)

    Args:
        eps (float, optional): term to improve numerical stability (default: 1e-7). See:
            https://github.com/pytorch/pytorch/issues/8069.

        reduction (string, optional): Specifies the reduction to apply to the output:
            ``'none'`` | ``'mean'`` | ``'sum'``. ``'none'``: no reduction will
            be applied, ``'mean'``: the weighted mean of the output is taken,
            ``'sum'``: the output will be summed. Default: ``'mean'``

    Shape:
        - Input: Shape :math:`(N, 3, 3)`.
        - Target: Shape :math:`(N, 3, 3)`.
        - Output: If :attr:`reduction` is ``'none'``, then :math:`(N)`. Otherwise, scalar.
    """

    def __init__(self, eps: float = 1e-7, reduction: str = "mean") -> None:
        super().__init__()
        self.eps = eps
        self.reduction = reduction

    def forward(self, input: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        R_diffs = input @ target.permute(0, 2, 1)
        # See: https://github.com/pytorch/pytorch/issues/7500#issuecomment-502122839.
        traces = R_diffs.diagonal(dim1=-2, dim2=-1).sum(-1)
        dists = torch.acos(torch.clamp((traces - 1) / 2, -1 + self.eps, 1 - self.eps))
        if self.reduction == "none":
            return dists
        elif self.reduction == "mean":
            return dists.mean()
        elif self.reduction == "sum":
            return dists.sum()
        


class VisualOdometry(pl.LightningModule):
    def __init__(self, dropout=0.0, learning_rate=1e-3):
        super().__init__()
        self.fnet = nn.Sequential(
            BasicConvEncoder(output_dim=256, norm_fn='instance', dropout=dropout),
            POLAUpdate(embed_dim=256, depth=6, num_head=8, window_size=7, neig_win_num=1)
        )
        self.conv1_position = nn.Conv2d(512, 16, kernel_size=1)
        # self.conv2_position = nn.Conv2d(256, 16, kernel_size=1)
        
        self.conv1_orientation = nn.Conv2d(512, 16, kernel_size=1)
        # self.conv2_orientation = nn.Conv2d(256, 16, kernel_size=1)
        
        # self.maxpool = nn.MaxPool2d(2)
        self.relu = nn.ReLU()
        
        self.fc1_position = nn.Linear(19200, 256)
        self.fc2_position = nn.Linear(256, 3)
        self.fc1_orientation = nn.Linear(19200, 256)
        self.fc2_orientation = nn.Linear(256, 4)
        
        self.geodesic = GeodesicLoss()
        self.mse = nn.MSELoss()

    def forward(self, image1, image2):
        image1 = 2 * (image1 / 255.0) - 1.0
        image2 = 2 * (image2 / 255.0) - 1.0

        fmap1, fmap2 = self.fnet([image1, image2])
        # print(fmap1.shape, fmap2.shape) #
        feats = torch.cat([fmap1, fmap2], dim=1)
        # print(feats.shape) #
        
        # Position pathway
        pos = self.conv1_position(feats)
        # pos = self.conv2_position(pos)
        pos = pos.view(pos.size(0), -1)  # Flatten
        pos = self.relu(self.fc1_position(pos))
        pos = self.fc2_position(pos)

        # Orientation pathway
        orient = self.conv1_orientation(feats)
        # orient = self.conv2_orientation(orient)
        orient = orient.view(orient.size(0), -1)  # Flatten
        orient = self.relu(self.fc1_orientation(orient))
        orient = self.fc2_orientation(orient)

        # Combine position and orientation into a single output tensor
        res = torch.cat([pos, orient], dim=1)
        
        return res
    
    def freeze_fnet(self):
        """Freeze the fnet layer to prevent updates during training."""
        for param in self.fnet.parameters():
            # print(param)
            param.requires_grad = False
    
    def custom_loss(self, y_hat, y):
        pos_hat, orient_hat = y_hat[:, :3], y_hat[:, 3:]
        pos, orient = y[:, :3], y[:, 3:]
        
        # print(y.shape, y_hat.shape)
        
        pos_loss = self.mse(pos_hat, pos)
        orient_loss = self.geodesic(quaternion_to_matrix(orient_hat[:, [-1, 0, 1, 2]]), quaternion_to_matrix(orient[:, [-1, 0, 1, 2]]))
        
        return pos_loss + orient_loss
        

    def training_step(self, batch, batch_idx):
        x, y = batch
        x1, x2 = x[:, 0], x[:, 1]
        y_hat = self(x1, x2)
        loss = self.custom_loss(y_hat, y)
        self.log('train_loss', loss, on_step=True, on_epoch=True, prog_bar=True, logger=True)
        return loss

    def validation_step(self, batch, batch_idx):
        x, y = batch
        x1, x2 = x[:, 0], x[:, 1]
        y_hat = self(x1, x2)
        loss = self.custom_loss(y_hat, y)
        self.log('val_loss', loss, on_step=True, on_epoch=True, prog_bar=True, logger=True)
        return loss

    def test_step(self, batch, batch_idx):
        x, y = batch
        x1, x2 = x[:, 0], x[:, 1]
        y_hat = self(x1, x2)
        loss = self.criterion(y_hat, y)
        self.log('test_loss', loss, on_step=True, on_epoch=True, prog_bar=True, logger=True)
        return loss
    

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=0.001)
        return optimizer
    
    
# class VOModel(nn.Module):
    
#     def __init__(self):
#         super().__init__()
        
#         # feature network
    
#         self.fnet = nn.Sequential(
#             BasicConvEncoder(output_dim=256, norm_fn='instance', dropout=args.dropout),
#             POLAUpdate(embed_dim=256, depth=6, num_head=8, window_size=7, neig_win_num=1)
#         )
        
        
        
#      def forward(self, image1, image2):
#         """ Estimate optical flow between pair of frames """

#         image1 = 2 * (image1 / 255.0) - 1.0
#         image2 = 2 * (image2 / 255.0) - 1.0

#         image1 = image1.contiguous()
#         image2 = image2.contiguous()

#         fmap1, fmap2 = self.fnet([image1, image2])

#         fmap1 = fmap1.float()
#         fmap2 = fmap2.float()

#         # # Self-attention update
#         print(fmap1.shape, fmap2.shape)

#         feats = torch.cat([fmap1, fmap2], dim=1)
#         print(feats.shape)

#         return feats
