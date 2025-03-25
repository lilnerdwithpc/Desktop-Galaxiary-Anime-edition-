import numpy as np
import random
import torch


class ReplayBuffer:
    def __init__(self, buffer_size: int, device=None):
        self.buffer_size = buffer_size
        self.buffer = []  # List of trajectories
        self.current_trajectory = []
        self.trajectory_count = 0
        self.is_full = False
        self.device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
    def add(self, state, action, reward, next_state, done, log_prob=None):
        """
        Add a transition to the current trajectory.
        For GRPO, we also store the log probability of the action.
        """
        transition = (state, action, reward, next_state, done)
        if log_prob is not None:
            transition = transition + (log_prob,)
        self.current_trajectory.append(transition)
        
        if done:
            if len(self.buffer) < self.buffer_size:
                self.buffer.append(self.current_trajectory)
                self.is_full = len(self.buffer) == self.buffer_size
            else:
                self.buffer[self.trajectory_count % self.buffer_size] = self.current_trajectory
                self.trajectory_count += 1
            
            self.current_trajectory = []
    
    def sample(self, batch_size):
        if not self.buffer:
            return []
            
        # Sample batch_size trajectories
        sampled_trajectories = random.sample(self.buffer, min(batch_size, len(self.buffer)))
        
        # Flatten if needed for training
        samples = []
        for trajectory in sampled_trajectories:
            samples.extend(trajectory)
            
        return samples
    
    def get_all_trajectories(self):
        """Return all trajectories in the buffer"""
        return self.buffer
    
    def clear(self):
        """Clear the buffer"""
        self.buffer = []
        self.current_trajectory = []
        self.trajectory_count = 0
        self.is_full = False
    
    def __len__(self):
        """Return the number of trajectories in the buffer"""
        return len(self.buffer)
    
    def get_trajectory_lengths(self):
        """Return the lengths of all trajectories in the buffer"""
        return [len(trajectory) for trajectory in self.buffer]
    
    def to_tensor_batch(self, batch):
        """Convert a batch of samples to tensors for GRPO training"""
        if not batch:
            return None
        
        # Determine if we have log_probs in our transitions
        has_log_probs = len(batch[0]) > 5
        
        states = torch.FloatTensor(np.array([s[0] for s in batch])).to(self.device)
        actions = torch.FloatTensor(np.array([s[1] for s in batch])).to(self.device)
        rewards = torch.FloatTensor(np.array([s[2] for s in batch])).to(self.device)
        next_states = torch.FloatTensor(np.array([s[3] for s in batch])).to(self.device)
        dones = torch.FloatTensor(np.array([s[4] for s in batch])).to(self.device)
        
        if has_log_probs:
            log_probs = torch.FloatTensor(np.array([s[5] for s in batch])).to(self.device)
            return states, actions, rewards, next_states, dones, log_probs
        
        return states, actions, rewards, next_states, dones