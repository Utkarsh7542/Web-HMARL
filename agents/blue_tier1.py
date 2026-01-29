"""
Blue Agent Tier 1: Detection Specialist - REBALANCED VERSION
✅ Smaller model (1 layer, 64 hidden units - down from 2 layers, 128 units)
✅ Added prediction uncertainty and noise
✅ Reduced overconfidence (confidence scaled down 20%)
✅ Better training stability with progressive difficulty
"""

import torch
import torch.nn as nn
import numpy as np
import os

class Tier1Detector(nn.Module):
    """
    Tier 1: Detection Specialist - REBALANCED
    Architecture: Single Bidirectional LSTM (1 layer, 64 hidden units) + Attention
    Input: 127-dimensional feature vectors
    Output: 4 actions (Pass, Escalate, Suggest Block, Immediate Block)
    
    CHANGES:
    - Reduced from 2 layers → 1 layer
    - Reduced from 128 hidden → 64 hidden
    - Smaller fully connected layers
    - More dropout for regularization
    """
    
    def __init__(self, input_dim=127, hidden_dim=64, num_layers=1, num_actions=4):
        super().__init__()
        
        self.input_dim = input_dim
        self.hidden_dim = 64    # REDUCED from 128
        self.num_layers = 1     # REDUCED from 2
        
        # Bidirectional LSTM (SMALLER)
        self.lstm = nn.LSTM(
            input_size=input_dim,
            hidden_size=64,      # REDUCED
            num_layers=1,        # REDUCED
            bidirectional=True,
            batch_first=True,
            dropout=0.0          # Only 1 layer, so no inter-layer dropout
        )
        
        # Attention mechanism (SMALLER)
        self.attention = nn.Sequential(
            nn.Linear(64 * 2, 32),   # REDUCED from 64
            nn.Tanh(),
            nn.Dropout(0.3),         # Added dropout
            nn.Linear(32, 1)
        )
        
        # Output layers (MUCH SMALLER)
        self.fc = nn.Sequential(
            nn.Linear(64 * 2, 32),   # REDUCED from 64
            nn.ReLU(),
            nn.Dropout(0.4),         # Increased dropout
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
    Tier 1 Agent Wrapper - REBALANCED
    Handles training, action selection, and threshold management
    
    CHANGES:
    - Reduced confidence scaling (now 20% reduction instead of 15%)
    - Added prediction noise
    - Slower learning rate
    - More conservative thresholds
    """
    
    def __init__(self, learning_rate=2e-4, device=None):  # REDUCED from 3e-4
        # Device
        self.device = device if device else torch.device(
            'cuda' if torch.cuda.is_available() else 'cpu'
        )
        
        # Model (SMALLER)
        self.model = Tier1Detector().to(self.device)
        
        # Optimizer (SLOWER learning)
        self.optimizer = torch.optim.Adam(
            self.model.parameters(), 
            lr=learning_rate
        )
        
        # Loss
        self.criterion = nn.CrossEntropyLoss()
        
        # Thresholds (MORE CONSERVATIVE)
        self.thresholds = {
            'pass': 0.35,       # Raised from 0.3 (less likely to pass)
            'escalate': 0.65,   # Raised from 0.6 (more escalation)
            'suggest': 0.92,    # Raised from 0.9 (harder to suggest block)
            'block': 0.92       # Raised from 0.9 (harder to block)
        }
        
        # Training metrics
        self.train_losses = []
        self.train_accuracies = []
        
        # Uncertainty parameters
        self.confidence_scale = 0.80    # Reduce confidence by 20%
        self.prediction_noise = 0.08    # Add noise to predictions
        
    def select_action(self, observation, deterministic=False):
        """
        Select action based on observation with UNCERTAINTY
        
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
        
        # ADD UNCERTAINTY: Reduce overconfidence
        confidence_adjusted = confidence.item() * self.confidence_scale
        
        # Add random noise to prevent perfect confidence
        noise = np.random.uniform(-self.prediction_noise, self.prediction_noise)
        confidence_adjusted += noise
        confidence_adjusted = np.clip(confidence_adjusted, 0.0, 1.0)
        
        if deterministic:
            return action.item(), confidence_adjusted
        else:
            # Sample from distribution with added noise for exploration
            probs_np = probs[0].cpu().numpy()
            
            # Add small noise to probabilities
            probs_np += np.random.dirichlet([0.3] * len(probs_np)) * 0.1
            probs_np = probs_np / probs_np.sum()  # Re-normalize
            
            action_sampled = np.random.choice(len(probs_np), p=probs_np)
            return action_sampled, probs_np[action_sampled]
    
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
            
            # Get malicious probability
            threat_score = probs[0, 3].item()
        
        # Apply MORE CONSERVATIVE thresholds
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
        Single training step (supervised learning)
        
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
        
        # Gradient clipping (prevent too fast learning)
        torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=0.5)
        
        self.optimizer.step()
        
        # Metrics
        with torch.no_grad():
            preds = torch.argmax(logits, dim=-1)
            accuracy = (preds == action_tensor).float().mean().item()
        
        self.train_losses.append(loss.item())
        self.train_accuracies.append(accuracy)
        
        return loss.item(), accuracy
    
    def add_training_noise(self, enable=True, noise_level=0.1):
        """
        Add noise during training to prevent overfitting
        
        Args:
            enable: Whether to add noise
            noise_level: How much noise (0.0-1.0)
        """
        if enable:
            self.prediction_noise = noise_level
        else:
            self.prediction_noise = 0.0
    
    def adjust_confidence_scaling(self, scale):
        """
        Adjust how much we scale down confidence
        
        Args:
            scale: float, multiply confidence by this (0.0-1.0)
        """
        self.confidence_scale = np.clip(scale, 0.0, 1.0)
    
    def save(self, path):
        """Save model checkpoint"""
        try:
            os.makedirs(os.path.dirname(path), exist_ok=True)
            
            torch.save({
                'model_state_dict': self.model.state_dict(),
                'optimizer_state_dict': self.optimizer.state_dict(),
                'thresholds': self.thresholds,
                'train_losses': self.train_losses,
                'train_accuracies': self.train_accuracies,
                'confidence_scale': self.confidence_scale,
                'prediction_noise': self.prediction_noise,
            }, path)
            
            if os.path.exists(path):
                file_size = os.path.getsize(path) / 1024
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
        
        if 'confidence_scale' in checkpoint:
            self.confidence_scale = checkpoint['confidence_scale']
        if 'prediction_noise' in checkpoint:
            self.prediction_noise = checkpoint['prediction_noise']
        
        print(f"Tier 1 model loaded from {path}")


# Test
if __name__ == "__main__":
    print("Testing Tier 1 Detector (Rebalanced)...")
    
    # Create agent
    agent = Tier1Agent()
    print(f"Using device: {agent.device}")
    print(f"Confidence scaling: {agent.confidence_scale}")
    print(f"Prediction noise: {agent.prediction_noise}")
    
    # Model size comparison
    total_params = sum(p.numel() for p in agent.model.parameters())
    print(f"Total parameters: {total_params:,} (much smaller than before)")
    
    # Test forward pass
    batch_size = 16
    test_obs = np.random.randn(batch_size, 127).astype(np.float32)
    
    # Test action selection with uncertainty
    print("\nTesting action selection (10 samples from same input):")
    single_obs = test_obs[0]
    actions = []
    confidences = []
    
    for _ in range(10):
        action, confidence = agent.select_action(single_obs, deterministic=False)
        actions.append(action)
        confidences.append(confidence)
    
    print(f"Actions: {actions}")
    print(f"Confidences: {[f'{c:.3f}' for c in confidences]}")
    print(f"Variation in actions: {len(set(actions))} different actions")
    print(f"Confidence range: [{min(confidences):.3f}, {max(confidences):.3f}]")
    
    # Test training step
    test_actions = np.random.randint(0, 4, size=batch_size)
    loss, acc = agent.train_step(test_obs, test_actions)
    print(f"\nTraining - Loss: {loss:.4f}, Accuracy: {acc:.3f}")
    
    # Test threshold-based selection
    action_thresh, score = agent.select_action_with_thresholds(test_obs[0])
    print(f"\nThreshold-based action: {action_thresh}, Threat score: {score:.3f}")
    
    # Test confidence adjustment
    print(f"\nTesting confidence adjustment:")
    print(f"Original confidence scale: {agent.confidence_scale}")
    agent.adjust_confidence_scaling(0.6)
    print(f"New confidence scale: {agent.confidence_scale}")
    
    action, conf = agent.select_action(single_obs, deterministic=True)
    print(f"Action with 60% confidence: {action}, Confidence: {conf:.3f}")
    
    print("\n✅ Tier 1 rebalanced test complete!")