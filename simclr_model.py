import torch
import torch.nn as nn
from torchvision import models
import torch.nn.functional as F

class SimCLR(nn.Module):
    def __init__(self, base_model="resnet50", projection_dim=128):
        super().__init__()

        if base_model == "resnet50":
            self.encoder = models.resnet50(weights=None)
            dim_mlp = self.encoder.fc.in_features
            self.encoder.fc = nn.Identity()

        self.projection_head = nn.Sequential(
            nn.Linear(dim_mlp, 2048),
            nn.ReLU(),
            nn.Linear(2048, projection_dim)
        )

    def forward(self, x):
        h = self.encoder(x)
        z = self.projection_head(h)
        z = F.normalize(z, dim=1)
        return h, z


def nt_xent_loss(z1, z2, temperature=0.5):
    batch_size = z1.shape[0]

    z = torch.cat([z1, z2], dim=0)
    sim = torch.mm(z, z.t()) / temperature

    labels = torch.arange(batch_size).repeat(2)
    labels = labels.to(z.device)

    mask = torch.eye(batch_size * 2).bool().to(z.device)
    sim = sim[~mask].view(batch_size * 2, -1)

    loss = nn.CrossEntropyLoss()(sim, labels)
    return loss
