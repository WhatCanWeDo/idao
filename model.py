from torch import nn
import timm 
from config import device
class IDAONet(nn.Module):
    def __init__(self):
        super().__init__()
        self.model = timm.create_model('tf_efficientnet_b1_ns', num_classes=0).to(device)
        self.head = nn.Sequential(
            nn.Linear(1280, 256),
            nn.ReLU()
        ).to(device)
        self.classifier = nn.Sequential(
            nn.Linear(1280, 256),
            nn.ReLU(),
            nn.Linear(256, 1),
            nn.Sigmoid()
        ).to(device)

        self.regressor = nn.Sequential(
            nn.Linear(1280, 512),
            nn.ReLU(),
            nn.Linear(512, 256),
            nn.ReLU(),
            nn.Linear(256, 1),
        ).to(device)
    def forward(self, x):
        x = self.model.forward(x)
        res_cl = self.classifier(x)
        res_reg = self.regressor(x)
        return res_cl, res_reg
