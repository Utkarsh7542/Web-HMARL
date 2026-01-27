"""
Red Agent: Adversarial Attacker
DQN-based agent that learns to evade blue defenses 
Explores 20 SQL injection variants and obfuscation techniques
"""

import torch
import torch.nn as nn
import numpy as np
from collections import deque
import random
import os

class DQNetwork(nn.Module):
    """
    Deep Q-Network for red agent
    Maps defense patterns (50-dim) to attack action values (20 actions)
    """
    
    def __init__(self, state_dim=50, action_dim=20, hidden_dim=256):
        super().__init__()
        
        self.network = nn.Sequential(
            nn.Linear(state_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 128),
            nn.ReLU(),
            nn.Linear(128, action_dim)
        )
    
    def forward(self, x):
        return self.network(x)


class RedAgent:
    """
    Red Agent: SQL Injection Attacker
    
    Architecture: DQN with experience replay
    State: Defense patterns (50-dim)
    Actions: 20 SQL injection variants
    Reward: +10 (bypass), +5 (novel), -2 (blocked), +1 (expensive analysis)
    """
    
    def __init__(self, state_dim=50, action_dim=20, learning_rate=1e-4, device=None):
        # Device
        self.device = device if device else torch.device(
            'cuda' if torch.cuda.is_available() else 'cpu'
        )
        
        # Q-networks
        self.q_network = DQNetwork(state_dim, action_dim).to(self.device)
        self.target_network = DQNetwork(state_dim, action_dim).to(self.device)
        self.target_network.load_state_dict(self.q_network.state_dict())
        
        # Optimizer
        self.optimizer = torch.optim.Adam(self.q_network.parameters(), lr=learning_rate)
        
        # Hyperparameters (Section 6.2)
        self.gamma = 0.99  # Discount factor
        self.epsilon = 0.9  # Start with high exploration
        self.epsilon_min = 0.1
        self.epsilon_decay = 0.995
        self.target_update_freq = 100
        self.steps = 0
        
        # Experience replay
        self.memory = deque(maxlen=10000)
        self.batch_size = 64
        
        # Attack success tracking (for novelty reward)
        self.attack_success_history = deque(maxlen=100)
        
        # Training metrics
        self.train_losses = []
        self.success_rates = []
        
    def select_action(self, state, deterministic=False):
        """
        Epsilon-greedy action selection
        
        Args:
            state: (50,) numpy array of defense patterns
            deterministic: If True, no exploration
        
        Returns:
            action: int (0-19)
        """
        # Exploration
        if not deterministic and random.random() < self.epsilon:
            return random.randint(0, 19)
        
        # Exploitation
        state_tensor = torch.FloatTensor(state).unsqueeze(0).to(self.device)
        
        self.q_network.eval()
        with torch.no_grad():
            q_values = self.q_network(state_tensor)
        
        return q_values.argmax().item()
    
    def store_transition(self, state, action, reward, next_state, done):
        """Store experience in replay buffer"""
        self.memory.append((state, action, reward, next_state, done))
    
    def train_step(self):
        """
        Sample from replay buffer and train
        Returns loss if training happened, None otherwise
        """
        if len(self.memory) < self.batch_size:
            return None
        
        # Sample batch
        batch = random.sample(self.memory, self.batch_size)
        states, actions, rewards, next_states, dones = zip(*batch)
        
        # Convert to tensors
        states = torch.FloatTensor(np.array(states)).to(self.device)
        actions = torch.LongTensor(actions).to(self.device)
        rewards = torch.FloatTensor(rewards).to(self.device)
        next_states = torch.FloatTensor(np.array(next_states)).to(self.device)
        dones = torch.FloatTensor(dones).to(self.device)
        
        # Current Q values: Q(s, a)
        current_q = self.q_network(states).gather(1, actions.unsqueeze(1)).squeeze()
        
        # Target Q values: r + γ * max_a' Q_target(s', a')
        with torch.no_grad():
            next_q = self.target_network(next_states).max(1)[0]
            target_q = rewards + (1 - dones) * self.gamma * next_q
        
        # Loss
        loss = nn.MSELoss()(current_q, target_q)
        
        # Optimize
        self.optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(self.q_network.parameters(), max_norm=1.0)
        self.optimizer.step()
        
        # Update target network
        self.steps += 1
        if self.steps % self.target_update_freq == 0:
            self.target_network.load_state_dict(self.q_network.state_dict())
        
        # Decay epsilon
        self.epsilon = max(self.epsilon_min, self.epsilon * self.epsilon_decay)
        
        self.train_losses.append(loss.item())
        return loss.item()
    
    def record_attack_result(self, success):
        """Track attack success for novelty rewards"""
        self.attack_success_history.append(success)
        
        # Calculate recent success rate
        if len(self.attack_success_history) >= 10:
            recent_success = sum(list(self.attack_success_history)[-10:]) / 10
            self.success_rates.append(recent_success)
    
    def get_success_rate(self):
        """Get recent attack success rate"""
        if len(self.attack_success_history) == 0:
            return 0.0
        return sum(self.attack_success_history) / len(self.attack_success_history)
    
    def save(self, path):
        """Save model checkpoint"""
        try:
            # Ensure directory exists
            os.makedirs(os.path.dirname(path), exist_ok=True)
            
            torch.save({
                'q_network_state_dict': self.q_network.state_dict(),
                'target_network_state_dict': self.target_network.state_dict(),
                'optimizer_state_dict': self.optimizer.state_dict(),
                'epsilon': self.epsilon,
                'steps': self.steps,
                'train_losses': self.train_losses,
                'success_rates': self.success_rates
            }, path)
            
            if os.path.exists(path):
                file_size = os.path.getsize(path) / 1024  # KB
                print(f"[OK] Red agent saved to {path} ({file_size:.1f} KB)")
            else:
                print(f"[WARN] Model file may not have been created at {path}")
        except Exception as e:
            print(f"[ERROR] Error saving Red agent: {e}")
    
    def load(self, path):
        """Load model checkpoint"""
        checkpoint = torch.load(path, map_location=self.device)
        self.q_network.load_state_dict(checkpoint['q_network_state_dict'])
        self.target_network.load_state_dict(checkpoint['target_network_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        self.epsilon = checkpoint['epsilon']
        self.steps = checkpoint['steps']
        print(f"Red agent loaded from {path}")


# Test
if __name__ == "__main__":
    print("Testing Red Agent (DQN)...")
    
    # Create agent
    agent = RedAgent()
    print(f"Using device: {agent.device}")
    print(f"Initial epsilon: {agent.epsilon}")
    
    # Test action selection
    test_state = np.random.randn(50).astype(np.float32)
    
    # Exploration
    action_explore = agent.select_action(test_state, deterministic=False)
    print(f"Action (with exploration): {action_explore}")
    
    # Exploitation
    action_exploit = agent.select_action(test_state, deterministic=True)
    print(f"Action (exploitation only): {action_exploit}")
    
    # Simulate experience collection
    print("\nSimulating experience collection...")
    for i in range(200):
        state = np.random.randn(50).astype(np.float32)
        action = agent.select_action(state)
        reward = np.random.choice([10, -2])  # Success or blocked
        next_state = np.random.randn(50).astype(np.float32)
        done = (i % 100 == 99)
        
        agent.store_transition(state, action, reward, next_state, done)
        agent.record_attack_result(reward > 0)
    
    print(f"Memory size: {len(agent.memory)}")
    print(f"Success rate: {agent.get_success_rate():.2%}")
    
    # Test training
    print("\nTraining for 10 steps...")
    for _ in range(10):
        loss = agent.train_step()
        if loss:
            print(f"Loss: {loss:.4f}, Epsilon: {agent.epsilon:.3f}")
    
    print(f"\nFinal epsilon: {agent.epsilon:.3f}")
    print(f"Target network updates: {agent.steps // agent.target_update_freq}")
    
    print("\n[OK] Red agent test complete!")
