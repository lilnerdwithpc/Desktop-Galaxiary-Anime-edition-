import torch
from torch import nn
import torch.nn.functional as F
import numpy as np
from gala_net import Network

"""
GRPO, or Group Relative Policy Optimization is a Reinfocement Learning algorithm modified upon PPO
Used in training DeepSeek-R1
GRPO completely removes the value network, training solely the policy network
GRPO still have the reward function, reference model
GRPO will have the policy generate/perform many sequence of actions, the reward function will then reward all the action sequences
The rewards will the be normalized, and all used to update the policy network
This directly shows how good the sequences of actions are "relatively" to each other
"""

class Agent():
    def __init__(self, num_joints:int, action_dim:int, action_space:int, weights_path:str=None, lr:float=0.0001, 
                 epsilon:float=0.2, beta:float=0.01, reference_model_path:str=None):
        self.model = Network(num_joints, action_dim, action_space)
        if weights_path:
            self.model.load_state_dict(torch.load(weights_path))
        
        # Initialize reference model if provided
        self.ref_model = None
        if reference_model_path:
            self.ref_model = Network(num_joints, action_dim, action_space)
            self.ref_model.load_state_dict(torch.load(reference_model_path))
            self.ref_model.eval()
            
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=lr)
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model.to(self.device)
        if self.ref_model:
            self.ref_model.to(self.device)
            
        self.action_dim = action_dim
        self.action_space = action_space
        self.num_joints = num_joints
        self.weights_path = weights_path
        self.lr = lr
        self.epsilon = epsilon  # Clipping parameter
        self.beta = beta        # KL divergence weight
        self.model.eval()
        
    def get_action(self, state:np.ndarray, prev_state:np.ndarray=None) -> tuple:
        """
        Get action from policy for given state.
        
        Args:
            state: Current state
            prev_state: Previous state (optional)
            
        Returns:
            Tuple of (action, new_prev_state, log_prob)
        """
        state = torch.from_numpy(state).float().unsqueeze(0).to(self.device)
        
        # Convert prev_state if provided
        if prev_state is not None:
            prev_state = torch.from_numpy(prev_state).float().unsqueeze(0).to(self.device)
        
        with torch.no_grad():
            # Use the sample_action method for continuous actions
            action, new_prev, log_prob = self.model.sample_action(state, prev_state)
        
        return (
            action.cpu().numpy().flatten(), 
            new_prev.cpu().numpy().flatten(),
            log_prob.cpu().numpy().flatten()
        )
        
    def save_weights(self, path:str):
        torch.save(self.model.state_dict(), path)
        
    def optimize(self, states:np.ndarray, actions:np.ndarray, rewards:np.ndarray, 
                 old_log_probs:np.ndarray, prev_states:np.ndarray=None) -> float:
        """
        Optimize policy network using GRPO algorithm.
        
        Args:
            states: Batch of states for a group of samples
            actions: Batch of actions taken for each state
            rewards: Rewards received for each action
            old_log_probs: Log probabilities of actions under old policy
            prev_states: Previous hidden states (optional)
            
        Returns:
            Loss value
        """
        # Convert to tensors
        states = torch.from_numpy(states).float().to(self.device)
        actions = torch.from_numpy(actions).float().to(self.device)
        rewards = torch.from_numpy(rewards).float().to(self.device)
        old_log_probs = torch.from_numpy(old_log_probs).float().to(self.device)
        
        if prev_states is not None:
            prev_states = torch.from_numpy(prev_states).float().to(self.device)
        
        # Calculate advantages as per GRPO algorithm
        # Normalize rewards within the group to get advantages
        advantages = (rewards - rewards.mean()) / (rewards.std() + 1e-8)
        
        # Set model to training mode
        self.model.train()
        
        # Get new log probabilities
        new_log_probs = self.model.get_log_prob(states, actions, prev_states)
        
        # Calculate policy ratio
        ratio = torch.exp(new_log_probs - old_log_probs)
        
        # Calculate surrogate loss with clipping
        clip_adv = torch.clamp(ratio, 1 - self.epsilon, 1 + self.epsilon) * advantages
        policy_loss = -torch.min(ratio * advantages, clip_adv).mean()
        
        # Calculate KL divergence term if reference model exists
        kl_loss = 0.0
        if self.ref_model:
            with torch.no_grad():
                ref_log_probs = self.ref_model.get_log_prob(states, actions, prev_states)
            
            # Calculate KL divergence as per formula
            # D_KL = p_ref/p_current * log(p_ref/p_current) - 1
            ratio_kl = torch.exp(ref_log_probs - new_log_probs)
            kl_div = ratio_kl * (ref_log_probs - new_log_probs) - 1
            kl_loss = self.beta * kl_div.mean()
        
        # Total loss
        loss = policy_loss + kl_loss
        
        # Perform optimization
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
        
        # Set model back to evaluation mode
        self.model.eval()
        
        return loss.item()
    
    def collect_trajectories(self, env, num_trajectories:int, max_steps:int) -> tuple:
        """
        Collect multiple trajectories by running the current policy in the environment.
        
        Args:
            env: Environment to collect trajectories from
            num_trajectories: Number of trajectories to collect
            max_steps: Maximum steps per trajectory
            
        Returns:
            Tuple of (states, actions, rewards, log_probs, prev_states)
        """
        all_states = []
        all_actions = []
        all_rewards = []
        all_log_probs = []
        all_prev_states = []
        
        for _ in range(num_trajectories):
            states = []
            actions = []
            rewards = []
            log_probs = []
            prev_states = []
            
            state = env.reset()
            prev_state = None
            done = False
            step = 0
            
            while not done and step < max_steps:
                # Get action from policy
                action, new_prev, log_prob = self.get_action(state, prev_state)
                
                # Take action in environment
                next_state, reward, done, _ = env.step(action)
                
                # Store trajectory data
                states.append(state)
                actions.append(action)
                rewards.append(reward)
                log_probs.append(log_prob)
                prev_states.append(prev_state if prev_state is not None else np.zeros(128))
                
                # Update for next step
                state = next_state
                prev_state = new_prev
                step += 1
            
            # Store completed trajectory
            all_states.append(states)
            all_actions.append(actions)
            all_rewards.append(rewards)
            all_log_probs.append(log_probs)
            all_prev_states.append(prev_states)
        
        # Convert to numpy arrays
        states_np = np.concatenate([np.array(s) for s in all_states])
        actions_np = np.concatenate([np.array(a) for a in all_actions])
        rewards_np = np.concatenate([np.array(r) for r in all_rewards])
        log_probs_np = np.concatenate([np.array(lp) for lp in all_log_probs])
        prev_states_np = np.concatenate([np.array(ps) for ps in all_prev_states])
        
        return states_np, actions_np, rewards_np, log_probs_np, prev_states_np