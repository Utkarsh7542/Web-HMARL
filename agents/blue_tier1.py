
import torch
import torch.nn as nn
import numpy as np
import os

class Tier1Detector(nn.Module):
    """
    Tier 1: Detection Specialist
    Architecture: Bidirectional LSTM (3 layers, 256 hidden units) + Attention
    Input: 127-dimensional feature vectors
    Output: 4 actions (Pass, Escalate, Suggest Block, Immediate Block)
    """
    
    def __init__(self, input_dim=127, hidden_dim=256, num_layers=3, num_actions=4):
        super().__init__()
        
        self.input_dim = input_dim
        self.hidden_dim = 128  # REDUCED from 256
        self.num_layers = 2     # REDUCED from 3
        
        # Bidirectional LSTM (SMALLER to make blue less powerful initially)
        self.lstm = nn.LSTM(
            input_size=input_dim,
            hidden_size=128,  # REDUCED
            num_layers=2,     # REDUCED
            bidirectional=True,
            batch_first=True,
            dropout=0.3 if num_layers > 1 else 0
        )
        
        # Attention mechanism
        self.attention = nn.Sequential(
            nn.Linear(128 * 2, 64),  # REDUCED
            nn.Tanh(),
            nn.Linear(64, 1)
        )
        
        # Output layers (SMALLER)
        self.fc = nn.Sequential(
            nn.Linear(128 * 2, 64),  # REDUCED
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(64, 32),       # REDUCED
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(32, num_actions)
        )
        
    def forward(self, x):
        """
        Forward pass
        Args:
            x: (batch, 127) feature vectors
        Returns:
            logits: (batch, 4) action logits
        """
        # Add sequence dimension: (batch, 127) -> (batch, 1, 127)
        x = x.unsqueeze(1)
        
        # LSTM: (batch, 1, 127) -> (batch, 1, hidden*2)
        lstm_out, _ = self.lstm(x)
        
        # Attention weights: (batch, 1, 1)
        attention_weights = torch.softmax(self.attention(lstm_out), dim=1)
        
        # Apply attention: (batch, hidden*2)
        attended = torch.sum(attention_weights * lstm_out, dim=1)
        
        # Output: (batch, 4)
        logits = self.fc(attended)
        
        return logits
    
    def predict_with_confidence(self, x):
        """
        Predict action with confidence score
        Returns: action, confidence, probabilities
        """
        logits = self.forward(x)
        probs = torch.softmax(logits, dim=-1)
        confidence, action = torch.max(probs, dim=-1)
        
        return action, confidence, probs


class Tier1Agent:
    """
    Tier 1 Agent Wrapper
    Handles training, action selection, and threshold management
    """
    
    def __init__(self, learning_rate=3e-4, device=None):
        # Device
        self.device = device if device else torch.device(
            'cuda' if torch.cuda.is_available() else 'cpu'
        )
        
        # Model
        self.model = Tier1Detector().to(self.device)
        
        # Optimizer
        self.optimizer = torch.optim.Adam(
            self.model.parameters(), 
            lr=learning_rate
        )
        
        # Loss
        self.criterion = nn.CrossEntropyLoss()
        
        # Thresholds (Section 6.3)
        self.thresholds = {
            'pass': 0.3,       # Score < 0.3: Pass directly
            'escalate': 0.6,   # 0.3-0.6: Escalate to Tier 2
            'suggest': 0.9,    # 0.6-0.9: Suggest block
            'block': 0.9       # > 0.9: Immediate block
        }
        
        # Training metrics
        self.train_losses = []
        self.train_accuracies = []
        
    def select_action(self, observation, deterministic=False):
        """
        Select action based on observation
        
        Args:
            observation: (127,) numpy array
            deterministic: If True, use argmax; else sample
        
        Returns:
            action: int (0-3)
            confidence: float
        """
        obs_tensor = torch.FloatTensor(observation).unsqueeze(0).to(self.device)
        
        self.model.eval()
        with torch.no_grad():
            action, confidence, probs = self.model.predict_with_confidence(obs_tensor)
        
        # ADD UNCERTAINTY: reduce overconfidence
        # Scale confidence down to make blue less certain
        confidence_adjusted = confidence.item() * 0.85  # Reduce by 15%
        
        # Add small random noise to prevent perfect confidence
        confidence_adjusted += np.random.uniform(-0.05, 0.05)
        confidence_adjusted = np.clip(confidence_adjusted, 0.0, 1.0)
        
        if deterministic:
            return action.item(), confidence.item()
        else:
            # Sample from distribution (for exploration)
            action_sampled = torch.multinomial(probs, 1).item()
            return action_sampled, probs[0, action_sampled].item()
    
    def select_action_with_thresholds(self, observation):
        """
        Select action using threshold-based decision
        Implements Section 6.3 Tier 1 decision policy
        """
        obs_tensor = torch.FloatTensor(observation).unsqueeze(0).to(self.device)
        
        self.model.eval()
        with torch.no_grad():
            logits = self.model(obs_tensor)
            probs = torch.softmax(logits, dim=-1)
            
            # Get malicious probability (assuming action 3 = block for malicious)
            # In practice, you'd train this to output threat score
            threat_score = probs[0, 3].item()  # Probability of "immediate block"
        
        # Apply thresholds
        if threat_score < self.thresholds['pass']:
            action = 0  # Pass
        elif threat_score < self.thresholds['escalate']:
            action = 1  # Escalate to Tier 2
        elif threat_score < self.thresholds['suggest']:
            action = 2  # Suggest block
        else:
            action = 3  # Immediate block
        
        return action, threat_score
    
    def train_step(self, observations, actions, rewards=None):
        """
        Single training step (supervised learning initially)
        
        Args:
            observations: (batch, 127)
            actions: (batch,) target actions
            rewards: Optional rewards for RL fine-tuning
        """
        self.model.train()
        
        # Convert to tensors
        obs_tensor = torch.FloatTensor(observations).to(self.device)
        action_tensor = torch.LongTensor(actions).to(self.device)
        
        # Forward pass
        logits = self.model(obs_tensor)
        loss = self.criterion(logits, action_tensor)
        
        # Backward pass
        self.optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
        self.optimizer.step()
        
        # Metrics
        with torch.no_grad():
            preds = torch.argmax(logits, dim=-1)
            accuracy = (preds == action_tensor).float().mean().item()
        
        self.train_losses.append(loss.item())
        self.train_accuracies.append(accuracy)
        
        return loss.item(), accuracy
    
    def save(self, path):
        """Save model checkpoint"""
        try:
            # Ensure directory exists
            os.makedirs(os.path.dirname(path), exist_ok=True)
            
            torch.save({
                'model_state_dict': self.model.state_dict(),
                'optimizer_state_dict': self.optimizer.state_dict(),
                'thresholds': self.thresholds,
                'train_losses': self.train_losses,
                'train_accuracies': self.train_accuracies
            }, path)
            
            if os.path.exists(path):
                file_size = os.path.getsize(path) / 1024  # KB
                print(f"[OK] Tier 1 model saved to {path} ({file_size:.1f} KB)")
            else:
                print(f"[WARN] Model file may not have been created at {path}")
        except Exception as e:
            print(f"[ERROR] Error saving Tier 1 model: {e}")
    
    def load(self, path):
        """Load model checkpoint"""
        checkpoint = torch.load(path, map_location=self.device)
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        self.thresholds = checkpoint['thresholds']
        print(f"Tier 1 model loaded from {path}")


# Test
if __name__ == "__main__":
    print("Testing Tier 1 Detector...")
    
    # Create agent
    agent = Tier1Agent()
    print(f"Using device: {agent.device}")
    
    # Test forward pass
    batch_size = 16
    test_obs = np.random.randn(batch_size, 127).astype(np.float32)
    
    # Test action selection
    action, confidence = agent.select_action(test_obs[0])
    print(f"Selected action: {action}, Confidence: {confidence:.3f}")
    
    # Test training step
    test_actions = np.random.randint(0, 4, size=batch_size)
    loss, acc = agent.train_step(test_obs, test_actions)
    print(f"Training - Loss: {loss:.4f}, Accuracy: {acc:.3f}")
    
    # Test threshold-based selection
    action_thresh, score = agent.select_action_with_thresholds(test_obs[0])
    print(f"Threshold-based action: {action_thresh}, Threat score: {score:.3f}")
    
    print("\n[OK] Tier 1 test complete!")
