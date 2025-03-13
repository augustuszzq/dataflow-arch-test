import torch.nn as nn
import torch.nn.functional as F

class MNISTModel(nn.Module):
    def __init__(self, hparams):
        super(MNISTModel, self).__init__()
        self.hparams = hparams
        self.fc1 = nn.Linear(28 * 28, hparams['model']['hidden_sizes'][0])
        self.fc2 = nn.Linear(hparams['model']['hidden_sizes'][0], hparams['model']['hidden_sizes'][1])
        self.fc3 = nn.Linear(hparams['model']['hidden_sizes'][1], 10)
        self.dropout = nn.Dropout(hparams['model']['dropout'])

    def forward(self, x):
        x = x.view(-1, 28 * 28)
        x = F.relu(self.fc1(x))
        x = self.dropout(x)
        x = F.relu(self.fc2(x))
        x = self.dropout(x)
        x = self.fc3(x)
        return F.log_softmax(x, dim=1)
