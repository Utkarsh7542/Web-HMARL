"""
Blue Agent Tier 2: Response Specialist 
Feedforward network (128→64) with PPO
Multi-objective decision making: security vs usability
"""

import torch
import torch.nn as nn
import numpy as np

class Tier2ResponseNetwork(nn.Module):
    """
    Tier 2: Response Specialist
    Feedforward NN: 128 → 64 neurons 
    Actions: Allow, Block, Sanitize, Throttle
    """
    
    def __init__(self, input_dim=135, hidden_dims=[128, 64], num_actions=4):
        super().__init__()
        
        # Input: 127 (query features) + 4 (Tier 1 output) + 4 (context)
        self.network = nn.Sequential(
            nn.Linear(input_dim, hidden_dims[0]),  # 128 neurons
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(hidden_dims[0], hidden_dims[1]),  # 64 neurons
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(hidden_dims[1], num_actions)
        )
        
        # Value head (for PPO)
        self.value_head = nn.Sequential(
            nn.Linear(input_dim, 64),
            nn.ReLU(),
            nn.Linear(64, 1)
        )
    
    def forward(self, x):
        """Returns action logits and value estimate"""
        action_logits = self.network(x)
        value = self.value_head(x)
        return action_logits, value


class Tier2Agent:
    """
    Tier 2 Response Agent
    Contextual decision-making with business logic
    """
    
    def __init__(self, learning_rate=3e-4, device=None):
        self.device = device if device else torch.device(
            'cuda' if torch.cuda.is_available() else 'cpu'
        )
        
        self.model = Tier2ResponseNetwork().to(self.device)
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=learning_rate)
        
    def select_action(self, observation, tier1_output, context):
        """
        Select action with context
        
        Args:
            observation: (127,) query features
            tier1_output: (4,) Tier 1 action probabilities
            context: (4,) [user_reputation, endpoint_criticality, system_load, recent_fp_rate]
        """
        # Combine inputs
        combined = np.concatenate([observation, tier1_output, context])
        obs_tensor = torch.FloatTensor(combined).unsqueeze(0).to(self.device)
        
        self.model.eval()
        with torch.no_grad():
            logits, value = self.model(obs_tensor)
            probs = torch.softmax(logits, dim=-1)
            action = torch.multinomial(probs, 1).item()
        
        return action, probs[0].cpu().numpy(), value.item()
    
    def save(self, path):
        torch.save({
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict()
        }, path)
        print(f"Tier 2 saved to {path}")
    
    def load(self, path):
        checkpoint = torch.load(path, map_location=self.device)
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        print(f"Tier 2 loaded from {path}")

# Test
if __name__ == "__main__":
    print("Testing Tier 2 Response Agent...")
    
    agent = Tier2Agent()
    print(f"Using device: {agent.device}")
    
    # Test inputs
    observation = np.random.randn(127).astype(np.float32)
    tier1_output = np.array([0.1, 0.2, 0.3, 0.4])  # Tier 1 probs
    context = np.array([0.5, 0.7, 0.3, 0.1])  # Context
    
    action, probs, value = agent.select_action(observation, tier1_output, context)
    
    print(f"Selected action: {action}")
    print(f"Action probs: {probs}")
    print(f"Value estimate: {value:.3f}")
    
    print("\n✅ Tier 2 test complete!")