"""
Blue Agent Tier 2: Response Specialist - IMPROVED WITH PPO
✅ Proper PPO training implementation
✅ Advantage estimation (GAE)
✅ Multiple training epochs
✅ Experience buffer for trajectory collection
✅ Safe, stable hyperparameters
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from collections import deque
import os

class Tier2ResponseNetwork(nn.Module):
    """
    Tier 2: Response Specialist
    Feedforward NN: 135 → 128 → 64 neurons 
    Actions: Allow, Block, Sanitize, Throttle
    
    IMPROVED: Separate actor and critic heads for PPO
    """
    
    def __init__(self, input_dim=135, hidden_dims=[128, 64], num_actions=4):
        super().__init__()
        
        # Shared feature extraction
        # Input: 127 (query features) + 4 (Tier 1 output) + 4 (context) = 135
        self.shared = nn.Sequential(
            nn.Linear(input_dim, hidden_dims[0]),  # 135 → 128
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(hidden_dims[0], hidden_dims[1]),  # 128 → 64
            nn.ReLU(),
            nn.Dropout(0.2),
        )
        
        # Actor head (policy)
        self.actor = nn.Sequential(
            nn.Linear(hidden_dims[1], 32),
            nn.ReLU(),
            nn.Linear(32, num_actions)
        )
        
        # Critic head (value function)
        self.critic = nn.Sequential(
            nn.Linear(hidden_dims[1], 32),
            nn.ReLU(),
            nn.Linear(32, 1)
        )
    
    def forward(self, x):
        """
        Forward pass
        Returns: action_logits, value
        """
        features = self.shared(x)
        action_logits = self.actor(features)
        value = self.critic(features)
        return action_logits, value
    
    def get_action_and_value(self, x, action=None):
        """
        Get action distribution and value estimate
        
        Args:
            x: input tensor
            action: if provided, compute log prob for this action
        
        Returns:
            action, log_prob, entropy, value
        """
        logits, value = self.forward(x)
        probs = torch.softmax(logits, dim=-1)
        dist = torch.distributions.Categorical(probs)
        
        if action is None:
            action = dist.sample()
        
        log_prob = dist.log_prob(action)
        entropy = dist.entropy()
        
        return action, log_prob, entropy, value


class PPOBuffer:
    """
    Experience buffer for PPO trajectory collection
    Stores (s, a, r, s', done, log_prob, value)
    """
    
    def __init__(self, max_size=2048):
        self.max_size = max_size
        self.clear()
    
    def clear(self):
        """Clear buffer"""
        self.observations = []
        self.actions = []
        self.rewards = []
        self.dones = []
        self.log_probs = []
        self.values = []
        self.advantages = []
        self.returns = []
    
    def add(self, obs, action, reward, done, log_prob, value):
        """Add experience to buffer"""
        self.observations.append(obs)
        self.actions.append(action)
        self.rewards.append(reward)
        self.dones.append(done)
        self.log_probs.append(log_prob)
        self.values.append(value)
    
    def compute_advantages(self, last_value, gamma=0.99, gae_lambda=0.95):
        """
        Compute advantages using Generalized Advantage Estimation (GAE)
        
        Args:
            last_value: value estimate for the final state
            gamma: discount factor
            gae_lambda: GAE lambda parameter
        """
        # Convert to numpy
        rewards = np.array(self.rewards)
        values = np.array(self.values + [last_value])
        dones = np.array(self.dones)
        
        # Compute advantages
        advantages = np.zeros_like(rewards)
        last_gae = 0
        
        for t in reversed(range(len(rewards))):
            if t == len(rewards) - 1:
                next_value = last_value
            else:
                next_value = values[t + 1]
            
            # TD error: δ_t = r_t + γ * V(s_{t+1}) - V(s_t)
            delta = rewards[t] + gamma * next_value * (1 - dones[t]) - values[t]
            
            # GAE: A_t = δ_t + γλ * A_{t+1}
            advantages[t] = last_gae = delta + gamma * gae_lambda * (1 - dones[t]) * last_gae
        
        # Compute returns: R_t = A_t + V(s_t)
        returns = advantages + values[:-1]
        
        self.advantages = advantages.tolist()
        self.returns = returns.tolist()
    
    def get(self):
        """Get all experiences as tensors"""
        return {
            'observations': np.array(self.observations, dtype=np.float32),
            'actions': np.array(self.actions, dtype=np.int64),
            'log_probs': np.array(self.log_probs, dtype=np.float32),
            'advantages': np.array(self.advantages, dtype=np.float32),
            'returns': np.array(self.returns, dtype=np.float32),
        }
    
    def __len__(self):
        return len(self.observations)


class Tier2Agent:
    """
    Tier 2 Response Agent with PPO Training
    Contextual decision-making with reinforcement learning
    
    IMPROVEMENTS:
    - Proper PPO implementation with clipped surrogate objective
    - GAE for advantage estimation
    - Multiple epochs per update
    - Experience buffer for trajectory collection
    - Entropy bonus for exploration
    """
    
    def __init__(self, learning_rate=3e-4, device=None):
        self.device = device if device else torch.device(
            'cuda' if torch.cuda.is_available() else 'cpu'
        )
        
        self.model = Tier2ResponseNetwork().to(self.device)
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=learning_rate)
        
        # PPO hyperparameters
        self.clip_epsilon = 0.2      # Clipping parameter for PPO
        self.value_coef = 0.5        # Value loss coefficient
        self.entropy_coef = 0.01     # Entropy bonus coefficient
        self.max_grad_norm = 0.5     # Gradient clipping
        self.n_epochs = 4            # Number of PPO epochs per update
        self.batch_size = 64         # Mini-batch size
        self.gamma = 0.99            # Discount factor
        self.gae_lambda = 0.95       # GAE lambda
        
        # Experience buffer
        self.buffer = PPOBuffer(max_size=2048)
        
        # Training metrics
        self.train_losses = []
        self.policy_losses = []
        self.value_losses = []
        self.entropies = []
        
    def select_action(self, observation, tier1_output, context, deterministic=False):
        """
        Select action with context
        
        Args:
            observation: (127,) query features
            tier1_output: (4,) Tier 1 action probabilities
            context: (4,) [user_reputation, endpoint_criticality, system_load, recent_fp_rate]
            deterministic: If True, select argmax action
        
        Returns:
            action: int
            action_probs: numpy array
            value: float
            log_prob: float (for training)
        """
        # Combine inputs
        combined = np.concatenate([observation, tier1_output, context])
        obs_tensor = torch.FloatTensor(combined).unsqueeze(0).to(self.device)
        
        self.model.eval()
        with torch.no_grad():
            logits, value = self.model(obs_tensor)
            probs = torch.softmax(logits, dim=-1)
            
            if deterministic:
                action = torch.argmax(probs, dim=-1).item()
                log_prob = torch.log(probs[0, action]).item()
            else:
                dist = torch.distributions.Categorical(probs)
                action = dist.sample().item()
                log_prob = dist.log_prob(torch.tensor(action).to(self.device)).item()
        
        return action, probs[0].cpu().numpy(), value.item(), log_prob
    
    def store_experience(self, observation, tier1_output, context, action, reward, done, log_prob, value):
        """
        Store experience in PPO buffer
        
        Args:
            observation: (127,) query features
            tier1_output: (4,) Tier 1 output
            context: (4,) context vector
            action: int
            reward: float
            done: bool
            log_prob: float
            value: float
        """
        combined = np.concatenate([observation, tier1_output, context])
        self.buffer.add(combined, action, reward, done, log_prob, value)
    
    def train_step(self):
        """
        Train using PPO algorithm
        Only trains when buffer has enough experiences
        
        Returns:
            dict with training metrics or None
        """
        if len(self.buffer) < self.batch_size:
            return None
        
        # Compute advantages with last value = 0 (assuming episode ended)
        self.buffer.compute_advantages(last_value=0.0, gamma=self.gamma, gae_lambda=self.gae_lambda)
        
        # Get all experiences
        data = self.buffer.get()
        
        # Convert to tensors
        obs = torch.FloatTensor(data['observations']).to(self.device)
        actions = torch.LongTensor(data['actions']).to(self.device)
        old_log_probs = torch.FloatTensor(data['log_probs']).to(self.device)
        advantages = torch.FloatTensor(data['advantages']).to(self.device)
        returns = torch.FloatTensor(data['returns']).to(self.device)
        
        # Normalize advantages
        advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)
        
        # Training metrics
        epoch_policy_losses = []
        epoch_value_losses = []
        epoch_entropies = []
        
        # Multiple epochs of PPO updates
        for epoch in range(self.n_epochs):
            # Mini-batch training
            indices = np.arange(len(obs))
            np.random.shuffle(indices)
            
            for start in range(0, len(obs), self.batch_size):
                end = start + self.batch_size
                batch_indices = indices[start:end]
                
                # Get batch
                batch_obs = obs[batch_indices]
                batch_actions = actions[batch_indices]
                batch_old_log_probs = old_log_probs[batch_indices]
                batch_advantages = advantages[batch_indices]
                batch_returns = returns[batch_indices]
                
                # Forward pass
                _, new_log_probs, entropy, values = self.model.get_action_and_value(
                    batch_obs, batch_actions
                )
                
                # PPO policy loss (clipped surrogate objective)
                ratio = torch.exp(new_log_probs - batch_old_log_probs)
                surr1 = ratio * batch_advantages
                surr2 = torch.clamp(ratio, 1 - self.clip_epsilon, 1 + self.clip_epsilon) * batch_advantages
                policy_loss = -torch.min(surr1, surr2).mean()
                
                # Value loss (MSE between predicted and actual returns)
                value_loss = F.mse_loss(values.squeeze(), batch_returns)
                
                # Entropy bonus (encourages exploration)
                entropy_loss = -entropy.mean()
                
                # Total loss
                loss = policy_loss + self.value_coef * value_loss + self.entropy_coef * entropy_loss
                
                # Optimize
                self.optimizer.zero_grad()
                loss.backward()
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.max_grad_norm)
                self.optimizer.step()
                
                # Track metrics
                epoch_policy_losses.append(policy_loss.item())
                epoch_value_losses.append(value_loss.item())
                epoch_entropies.append(entropy.mean().item())
        
        # Average metrics across epochs
        avg_policy_loss = np.mean(epoch_policy_losses)
        avg_value_loss = np.mean(epoch_value_losses)
        avg_entropy = np.mean(epoch_entropies)
        
        self.policy_losses.append(avg_policy_loss)
        self.value_losses.append(avg_value_loss)
        self.entropies.append(avg_entropy)
        self.train_losses.append(avg_policy_loss + avg_value_loss)
        
        # Clear buffer after training
        self.buffer.clear()
        
        return {
            'policy_loss': avg_policy_loss,
            'value_loss': avg_value_loss,
            'entropy': avg_entropy,
            'total_loss': self.train_losses[-1]
        }
    
    def save(self, path):
        """Save model checkpoint"""
        try:
            os.makedirs(os.path.dirname(path), exist_ok=True)
            
            torch.save({
                'model_state_dict': self.model.state_dict(),
                'optimizer_state_dict': self.optimizer.state_dict(),
                'train_losses': self.train_losses,
                'policy_losses': self.policy_losses,
                'value_losses': self.value_losses,
                'entropies': self.entropies,
            }, path)
            
            if os.path.exists(path):
                file_size = os.path.getsize(path) / 1024
                print(f"[OK] Tier 2 (PPO) saved to {path} ({file_size:.1f} KB)")
            else:
                print(f"[WARN] Tier 2 file may not have been created at {path}")
        except Exception as e:
            print(f"[ERROR] Error saving Tier 2: {e}")
    
    def load(self, path):
        """Load model checkpoint"""
        checkpoint = torch.load(path, map_location=self.device)
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        
        if 'train_losses' in checkpoint:
            self.train_losses = checkpoint['train_losses']
        if 'policy_losses' in checkpoint:
            self.policy_losses = checkpoint['policy_losses']
        
        print(f"Tier 2 (PPO) loaded from {path}")


# Test
if __name__ == "__main__":
    print("Testing Tier 2 Response Agent (with PPO)...")
    
    agent = Tier2Agent()
    print(f"Using device: {agent.device}")
    print(f"PPO parameters:")
    print(f"  Clip epsilon: {agent.clip_epsilon}")
    print(f"  Epochs per update: {agent.n_epochs}")
    print(f"  Batch size: {agent.batch_size}")
    
    # Model size
    total_params = sum(p.numel() for p in agent.model.parameters())
    print(f"Total parameters: {total_params:,}")
    
    # Test inputs
    observation = np.random.randn(127).astype(np.float32)
    tier1_output = np.array([0.1, 0.2, 0.3, 0.4], dtype=np.float32)
    context = np.array([0.5, 0.7, 0.3, 0.1], dtype=np.float32)
    
    # Test action selection
    action, probs, value, log_prob = agent.select_action(
        observation, tier1_output, context, deterministic=False
    )
    
    print(f"\nAction selection:")
    print(f"  Action: {action}")
    print(f"  Probs: {probs}")
    print(f"  Value: {value:.3f}")
    print(f"  Log prob: {log_prob:.3f}")
    
    # Simulate trajectory collection
    print(f"\nSimulating trajectory collection...")
    for step in range(128):
        obs = np.random.randn(127).astype(np.float32)
        t1_out = np.random.rand(4).astype(np.float32)
        ctx = np.random.rand(4).astype(np.float32)
        
        action, _, value, log_prob = agent.select_action(obs, t1_out, ctx)
        
        # Simulate reward
        reward = np.random.randn()  # Random reward for testing
        done = (step % 50 == 49)  # Episode ends every 50 steps
        
        agent.store_experience(obs, t1_out, ctx, action, reward, done, log_prob, value)
    
    print(f"Buffer size: {len(agent.buffer)}")
    
    # Test training
    print(f"\nTraining PPO...")
    metrics = agent.train_step()
    
    if metrics:
        print(f"Training metrics:")
        print(f"  Policy loss: {metrics['policy_loss']:.4f}")
        print(f"  Value loss: {metrics['value_loss']:.4f}")
        print(f"  Entropy: {metrics['entropy']:.4f}")
        print(f"  Total loss: {metrics['total_loss']:.4f}")
    
    print(f"\nBuffer size after training: {len(agent.buffer)}")
    
    print("\n✅ Tier 2 PPO test complete!")