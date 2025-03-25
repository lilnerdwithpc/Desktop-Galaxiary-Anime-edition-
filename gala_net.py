import torch
import torchinfo
from torch import nn
import torch.nn.functional as F
from torch import tensor
import numpy as np

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
        self.mean = nn.Linear(hidden_size, action_space)
        self.log_std = nn.Linear(hidden_size, action_space)
        self.min_log_std = -20
        self.max_log_std = 2
        
    def forward(self, x:tensor) -> tuple:
        x = self.fc1(x)
        x = F.relu(x)
        
        # Output mean and log_std for continuous actions
        mean = self.mean(x)
        log_std = self.log_std(x)
        
        # Clamp log_std for numerical stability
        log_std = torch.clamp(log_std, self.min_log_std, self.max_log_std)
        std = torch.exp(log_std)
        
        return mean, std

class Network(nn.Module):
    """
    The neural network that would control the rig of Desktop Galaxiary
    """
    def __init__(self, num_joints:int, action_dim:int, action_space:int):
        super(Network, self).__init__()
        self.num_joints = num_joints
        self.action_dim = action_dim
        self.action_space = action_space

        self.fc1 = nn.Linear(num_joints, 64)
        self.fc2 = nn.Linear(64, 128)
        self.hist = Historian(256, 128, 128)

        self.fc_action = nn.Linear(action_dim, 128)
        self.fc_action2 = nn.Linear(128, 128)

        self.policy = Policy(256, 128, action_space)

    def forward(self, x:tensor, prev:tensor=None, action:tensor=None) -> tensor:
        if prev is None:
            prev = torch.zeros(x.size(0), 128, device=x.device)
        if action is None:
            action = torch.zeros(x.size(0), self.action_dim, device=x.device)
            
        x = self.fc1(x)
        x = F.relu(x)
        x = self.fc2(x)
        x = F.relu(x)
        x = torch.cat((x, prev), dim=1)
        #region
        #                                        ,_     _
        #  omg cat                               |\\_,-~/
        #          _..---...,""-._     ,/}/)     / _  _ |    ,--.
        #       .''        ,      ``..'(/-<     (  @  @ )   / ,-'
        #      /   _      {      )         \     \  _T_/-._( (
        #     ;   _ `.     `.   <         a(     /         `. \
        #   ,'   ( \  )      `.  \ __.._ .: y   |         _  \ |            cat.
        #  (  <\_-) )'-.____...\  `._   //-'s    \ \ ,  /      |
        #   `. `-' /-._)))      `-._)))           || |-_\__   /
        #     `...'                              ((_/`(____,-'
        #  omg torch.cat() 
        #
        #endregion
        x = self.hist(x.unsqueeze(1)) # Add unsqueeze to make it 3D
        h = self.fc_action(action)
        h = F.relu(h)
        h = self.fc_action2(h)
        h = F.relu(h)
        z = torch.cat((x, h), dim=1)
        
        # Get mean and std from policy
        mean, std = self.policy(z)
        
        # If in training mode, sample from distribution
        if self.training:
            # Create normal distribution
            dist = torch.distributions.Normal(mean, std)
            # Sample actions
            actions = dist.rsample()  # Using rsample() for reparameterization trick
        else:
            # In evaluation, just use the mean
            actions = mean
            
        return actions, x, mean, std
    
    def get_log_prob(self, states, actions, prev_states=None):
        """
        Calculate log probabilities of continuous actions given states.
        
        Args:
            states: Batch of states
            actions: Batch of continuous actions to calculate log probabilities for
            prev_states: Previous states (optional)
            
        Returns:
            Log probabilities of actions
        """
        if prev_states is None:
            prev_states = torch.zeros(states.size(0), 128, device=states.device)
            
        # Get mean and std from policy
        _, _, mean, std = self.forward(states, prev_states)
        
        # Create normal distribution
        dist = torch.distributions.Normal(mean, std)
        
        # Calculate log probabilities
        # For multivariate normal with diagonal covariance, we sum log probs across dimensions
        log_probs = dist.log_prob(actions).sum(dim=-1)
        
        return log_probs
    
    def sample_action(self, state, prev_state=None):
        """
        Sample an action from the policy.
        
        Args:
            state: Current state
            prev_state: Previous state (optional)
            
        Returns:
            Sampled action, new previous state, log probability
        """
        # Convert to tensor if numpy
        if isinstance(state, np.ndarray):
            state = torch.from_numpy(state).float()
        if len(state.shape) == 1:
            state = state.unsqueeze(0)
            
        if prev_state is not None and isinstance(prev_state, np.ndarray):
            prev_state = torch.from_numpy(prev_state).float()
            if len(prev_state.shape) == 1:
                prev_state = prev_state.unsqueeze(0)
        
        # Set to eval mode temporarily to ensure consistent behavior
        training_mode = self.training
        self.eval()
        
        with torch.no_grad():
            # Get policy parameters
            _, new_prev, mean, std = self.forward(state, prev_state)
            
            # Create distribution
            dist = torch.distributions.Normal(mean, std)
            
            # Sample action
            action = dist.sample()
            
            # Get log probability
            log_prob = dist.log_prob(action).sum(dim=-1)
        
        # Restore original training mode
        self.train(training_mode)
        
        return action, new_prev, log_prob

model = Network(12, 5, 3)

torchinfo.summary(model)

dummy_input = torch.rand(1, 12)
dummy_recu = torch.rand(1,128)
dummy_action = torch.rand(1, 5)
output = model(dummy_input, dummy_recu, dummy_action)
print(output[0].size(), output[1].size())
