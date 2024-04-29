import torch
from torch import nn
import pytorch_lightning as pl
from torch.utils.data import DataLoader

#Quaternion difference of two unit quaternions
def quat_norm_diff(q_a, q_b):
    assert(q_a.shape == q_b.shape)
    assert(q_a.shape[-1] == 4)
    if q_a.dim() < 2:
        q_a = q_a.unsqueeze(0)
        q_b = q_b.unsqueeze(0)
    return torch.min((q_a-q_b).norm(dim=1), (q_a+q_b).norm(dim=1)).squeeze()

def quat_squared_loss(q, q_target, reduce=True):
    assert(q.shape == q_target.shape)
    d = quat_norm_diff(q, q_target)
    losses =  0.5*d*d
    loss = losses.mean() if reduce else losses
    return loss

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
        


class BidirectionalLSTM(pl.LightningModule):
    def __init__(self, input_dim=6, hidden_dim=256, output_dim=7, num_layers=2, dropout_rate=0.25):
        super(BidirectionalLSTM, self).__init__()
        self.lstm = nn.LSTM(input_size=input_dim,
                            hidden_size=hidden_dim,
                            num_layers=num_layers,
                            dropout=dropout_rate,
                            bidirectional=True,
                            batch_first=True)
        
        self.out_layer = nn.Linear(hidden_dim * 2, output_dim)
        # self.out_layer = nn.Linear(hidden_dim * 2, hidden_dim)
        
        self.fc1 = nn.Linear(hidden_dim*2, 128)  # Multiply hidden_size by 2 for bidirectional
        self.fc2 = nn.Linear(128, output_dim)
        
        # Additional linear layer
        # self.final_layer = nn.Linear(hidden_dim, output_dim)
        self.relu = nn.ReLU()
        self.position_loss = nn.L1Loss()
        self.geodesic = GeodesicLoss()

    def forward(self, x):
        # print(x.shape)
        lstm_out, _ = self.lstm(x)
        last_time_step = lstm_out[:, -1, :]
        # output = self.out_layer(last_time_step)
        x = self.relu(self.fc1(last_time_step))
        output = self.fc2(x)
        return output
    
    def custom_loss(self, y_hat, y):
        pos_hat, orient_hat = y_hat[:, :3], y_hat[:, 3:]
        pos, orient = y[:, :3], y[:, 3:]
        # print(pos, orient)
        
        # print(y.shape, y_hat.shape)
        orient_hat = torch.nn.functional.normalize(orient_hat, p=2, dim=1)
        orient = torch.nn.functional.normalize(orient, p=2, dim=1)
        
        pos_loss = self.position_loss(pos_hat, pos)
        
        # print(orient_hat.shape, orient.shape)
        orient_loss = self.geodesic(quaternion_to_matrix(orient_hat), quaternion_to_matrix(orient))
        # print(f"Pos Loss: {pos_loss} and Orient Loss: {orient_loss}")
        
        return pos_loss + orient_loss

    def training_step(self, batch, batch_idx):
        x, y = batch
        # print(x.shape, y.shape)
        y_hat = self(x)
        loss = self.custom_loss(y_hat, y)
        self.log('train_loss', loss, on_step=True, on_epoch=True, prog_bar=True, logger=True)
        return loss

    def validation_step(self, batch, batch_idx):
        x, y = batch
        y_hat = self(x)
        loss = self.custom_loss(y_hat, y)
        # print(f"Y_hat: {y_hat} and Y: {y} and Loss: {loss}")
        self.log('val_loss', loss, on_step=True, on_epoch=True, prog_bar=True, logger=True)
        return loss

    def test_step(self, batch, batch_idx):
        x, y = batch
        y_hat = self(x)
        loss = self.custom_loss(y_hat, y)
        self.log('test_loss', loss, on_step=True, on_epoch=True, prog_bar=True, logger=True)
        return loss
    
    def epoch_end(self, outputs, phase):
        avg_loss = torch.stack([x for x in outputs]).mean()
        self.log(f'{phase}_loss', avg_loss)

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=0.001)
        return optimizer
