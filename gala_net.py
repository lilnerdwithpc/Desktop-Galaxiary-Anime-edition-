import torch
import torchinfo
from torch import nn
import torch.nn.functional as F
from torch import tensor

class RNN(nn.Module):
    """
    Recurrent Neural Network
    """
    def __init__(self, input_size:int, hidden_size:int, output_size:int):
        super(RNN, self).__init__()
        self.hidden_size = hidden_size
        self.rnn = nn.RNN(input_size, hidden_size, batch_first=True)
        self.fc = nn.Linear(hidden_size, output_size)

    def forward(self, x:tensor) -> tensor:
        out, _ = self.rnn(x)
        print(out.size())
        out = self.fc(out[:, -1, :])
        return out

class Historian(nn.Module):
    def __init__(self, input_size:int, hidden_size:int, output_size:int):
        super().__init__()
        self.lstm = nn.LSTM(input_size, hidden_size, batch_first=True)
        self.fc = nn.Linear(hidden_size, output_size)
        
    def forward(self, x:tensor) -> tensor:
        # Handle both 2D and 3D inputs:
        out, _ = self.lstm(x)
        # Handle both 2D and 3D outputs
        if out.dim() == 2:
            out = self.fc(out)
        else:
            out = self.fc(out[:, -1, :])
        return out

class Policy(nn.Module):
    def __init__(self, input_size:int, hidden_size:int, action_space:int):
        super().__init__()
        self.fc1 = nn.Linear(input_size, hidden_size)
        self.fc2 = nn.Linear(hidden_size, action_space)
        
    def forward(self, x:tensor) -> tensor:
        x = self.fc1(x)
        x = F.relu(x)
        x = self.fc2(x)
        x = F.sigmoid(x)
        return x

class Network(nn.Module):
    """
    The neural network that would control the rig of Desktop Galaxiary
    """
    def __init__(self, num_joints:int, action_dim:int, action_space:int):
        super(Network, self).__init__()
        self.num_joints = num_joints
        self.action_dim = action_dim

        self.fc1 = nn.Linear(num_joints, 64)
        self.fc2 = nn.Linear(64, 128)
        self.hist = Historian(256, 128, 128)

        self.fc_action = nn.Linear(action_dim, 128)
        self.fc_action2 = nn.Linear(128, 128)

        self.policy = Policy(256, 128, action_space)

    def forward(self, x:tensor, prev:tensor, action:tensor) -> tensor:
        x = self.fc1(x)
        x = F.relu(x)
        x = self.fc2(x)
        x = F.relu(x)
        x = torch.cat((x, prev), dim=1)
        x = self.hist(x.unsqueeze(1)) # Add unsqueeze to make it 3D
        h = self.fc_action(action)
        h = F.relu(h)
        h = self.fc_action2(h)
        h = F.relu(h)
        z = torch.cat((x, h), dim=1)
        p = self.policy(z)
        return p, x
        

model = Network(12, 5, 3)

torchinfo.summary(model)

dummy_input = torch.rand(1, 12)
dummy_recu = torch.rand(1,128)
dummy_action = torch.rand(1, 5)
output = model(dummy_input, dummy_recu, dummy_action)
print(output[0].size(), output[1].size())