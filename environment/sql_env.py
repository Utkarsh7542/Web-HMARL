"""
SQL Injection RL Environment
Gymnasium environment for red-blue adversarial training
Implements Markov Game formulation from Section 5.1
"""

import gymnasium as gym
from gymnasium import spaces
import numpy as np
import sys
from pathlib import Path

# Add project root to path for imports
_project_root = Path(__file__).parent.parent
if str(_project_root) not in sys.path:
    sys.path.insert(0, str(_project_root))

class SQLInjectionEnv(gym.Env):
    """
    Multi-agent environment for SQL injection attack/defense
    
    Agents:
    - Red: Generates SQL injection attacks
    - Blue: Detects and responds to attacks
    
    State: Query features (127-dim) + defense history
    Actions: Red=attack type, Blue=response decision
    Rewards: Zero-sum game (Section 6.5)
    """
    
    metadata = {'render_modes': ['human']}
    
    def __init__(self, dataset, dataset_loader=None, max_steps=1000):
        super().__init__()
        
        self.dataset = dataset  # DataFrame with queries
        self.dataset_loader = dataset_loader  # SQLInjectionDataset object for feature extraction
        self.max_steps = max_steps
        self.current_step = 0
        
        # Action spaces (Section 5.1)
        self.red_action_space = spaces.Discrete(20)  # 20 attack variants
        self.blue_action_space = spaces.Discrete(4)  # Allow, Block, Sanitize, Throttle
        
        # Observation spaces
        self.red_observation_space = spaces.Box(
            low=0, high=1, shape=(50,), dtype=np.float32
        )  # Red observes defense patterns
        
        self.blue_observation_space = spaces.Box(
            low=0, high=10, shape=(127,), dtype=np.float32
        )  # Blue observes query features (Tier 1 input)
        
        # State tracking
        self.reset()
    
    def reset(self, seed=None, options=None):
        """Reset environment for new episode"""
        super().reset(seed=seed)
        
        self.current_step = 0
        self.attack_history = []
        self.defense_history = []
        self.episode_stats = {
            'attacks_successful': 0,
            'attacks_blocked': 0,
            'false_positives': 0,
            'true_negatives': 0
        }
        
        # Shuffle dataset for new episode
        self.dataset_indices = np.random.permutation(len(self.dataset))
        self.current_idx = 0
        
        # Get first observation
        obs = self._get_observations()
        info = {}
        
        return obs, info
    
    def step(self, actions):
        """
        Execute one step of adversarial interaction
        
        Args:
            actions: dict with 'red' and 'blue' keys
        
        Returns:
            observations, rewards, terminated, truncated, info
        """
        red_action = actions['red']
        blue_action = actions['blue']
        
        # Get current query
        query_idx = self.dataset_indices[self.current_idx]
        query_data = self.dataset.iloc[query_idx]
        
        is_attack = query_data['label'] == 1
        
        # Blue decision (Section 6.3 Tier 2 actions)
        # 0=Allow, 1=Block, 2=Sanitize, 3=Throttle
        blue_blocks = blue_action in [1, 2, 3]  # Block, Sanitize, or Throttle
        
        # Determine outcome
        attack_successful = is_attack and not blue_blocks
        false_positive = not is_attack and blue_blocks
        true_positive = is_attack and blue_blocks
        true_negative = not is_attack and not blue_blocks
        
        # Calculate rewards (Section 6.5 reward functions)
        red_reward = self._calculate_red_reward(
            attack_successful, blue_blocks, red_action
        )
        blue_reward = self._calculate_blue_reward(
            true_positive, false_positive, true_negative, blue_action
        )
        
        # Update statistics
        if attack_successful:
            self.episode_stats['attacks_successful'] += 1
        if true_positive:
            self.episode_stats['attacks_blocked'] += 1
        if false_positive:
            self.episode_stats['false_positives'] += 1
        if true_negative:
            self.episode_stats['true_negatives'] += 1
        
        # Update history
        self.attack_history.append({
            'action': red_action,
            'success': attack_successful,
            'type': query_data['attack_type']
        })
        
        self.defense_history.append({
            'action': blue_action,
            'blocked': blue_blocks,
            'correct': true_positive or true_negative,
            'confidence': np.random.uniform(0.5, 1.0)  # Placeholder
        })
        
        # Move to next step
        self.current_step += 1
        self.current_idx = (self.current_idx + 1) % len(self.dataset)
        
        # Check if episode is done
        terminated = self.current_step >= self.max_steps
        truncated = False
        
        # Get next observations
        obs = self._get_observations() if not terminated else self._get_zero_observations()
        
        # Info dict
        info = {
            'attack_successful': attack_successful,
            'false_positive': false_positive,
            'true_positive': true_positive,
            'is_attack': is_attack,
            'query_type': query_data['attack_type'],
            'episode_stats': self.episode_stats.copy()
        }
        
        # Package for multi-agent
        observations = {'red': obs['red'], 'blue': obs['blue']}
        rewards = {'red': red_reward, 'blue': blue_reward}
        terminateds = {'red': terminated, 'blue': terminated}
        truncateds = {'red': truncated, 'blue': truncated}
        
        return observations, rewards, terminateds, truncateds, info
    
    def _calculate_red_reward(self, success, blocked, action):
        """
        Red agent reward (Section 6.2) - WITH NaN PROTECTION
        R_adversarial = +10 (bypass) +5 (novel) -2 (blocked) +1 (expensive analysis)
        """
        reward = 0.0
        
        if success:
            reward += 10.0  # Successfully bypassed defense
        
        if blocked:
            reward -= 2.0  # Got detected
        
        # Novelty bonus with proper safeguards
        if len(self.attack_history) > 0:
            recent_actions = [h['action'] for h in self.attack_history[-10:]]
            if len(recent_actions) > 0 and recent_actions.count(action) <= 2:
                reward += 1.0  # Exploring diverse attacks
        
        # NaN protection
        if np.isnan(reward) or np.isinf(reward):
            print(f"⚠️ WARNING: Invalid red reward: {reward}, resetting to 0.0")
            reward = 0.0
        
        return float(reward)

    def _calculate_blue_reward(self, tp, fp, tn, action):
        """
        Blue agent reward (Section 6.3 Tier 2) - WITH NaN PROTECTION
        R_response = 5×(blocks) - 3×(FP) - 2×(UX friction) + 1×(intel)
        """
        reward = 0.0
        
        if tp:
            reward += 5.0  # Correctly blocked attack
        
        if fp:
            reward -= 3.0  # False positive (user friction)
        
        if tn:
            reward += 1.0  # Correctly allowed benign
        
        # Action cost (some actions more expensive)
        if action == 2:  # Sanitize
            reward -= 0.5  # Small cost for processing
        elif action == 3:  # Throttle
            reward -= 1.0  # Higher cost for rate limiting
        
        # NaN protection
        if np.isnan(reward) or np.isinf(reward):
            print(f"⚠️ WARNING: Invalid blue reward: {reward}, resetting to 0.0")
            reward = 0.0
        
        return float(reward)
    
    def _get_observations(self):
        """Get current observations for both agents"""
        # Get current query
        query_idx = self.dataset_indices[self.current_idx]
        query_data = self.dataset.iloc[query_idx]
        
        # Blue observes query features (127-dim from Tier 1)
        # Prefer precomputed features if present (much faster than per-step extraction)
        if 'features' in query_data and query_data['features'] is not None:
            blue_obs = np.asarray(query_data['features'], dtype=np.float32)
        elif self.dataset_loader is not None:
            blue_obs = self.dataset_loader.extract_features(query_data['query'])
        else:
            # Fallback: create a temporary loader if not provided
            from data.sqli_dataset import SQLInjectionDataset
            loader = SQLInjectionDataset()
            blue_obs = loader.extract_features(query_data['query'])
        
        # Red observes defense patterns (50-dim)
        red_obs = self._get_red_observation()
        
        return {'red': red_obs, 'blue': blue_obs}
    
    def _get_zero_observations(self):
        """Return zero observations for terminal state"""
        return {
            'red': np.zeros(50, dtype=np.float32),
            'blue': np.zeros(127, dtype=np.float32)
        }
    
    def _get_red_observation(self):
        """
        Red agent observes defense patterns
        50-dim: block rates, accuracy, action distribution
        """
        obs = []
        
        if len(self.defense_history) == 0:
            return np.zeros(50, dtype=np.float32)
        
        recent = self.defense_history[-20:]  # Last 20 decisions
        
        # Block rate
        block_rate = sum(1 for d in recent if d['blocked']) / len(recent)
        obs.append(block_rate)
        
        # Accuracy rate
        accuracy = sum(1 for d in recent if d['correct']) / len(recent)
        obs.append(accuracy)
        
        # Average confidence
        avg_confidence = np.mean([d['confidence'] for d in recent])
        obs.append(avg_confidence)
        
        # Action distribution (4 actions)
        for action_id in range(4):
            action_freq = sum(1 for d in recent if d['action'] == action_id) / len(recent)
            obs.append(action_freq)
        
        # Attack success rate (from red's perspective)
        if len(self.attack_history) > 0:
            recent_attacks = self.attack_history[-20:]
            success_rate = sum(1 for a in recent_attacks if a['success']) / len(recent_attacks)
            obs.append(success_rate)
        else:
            obs.append(0.0)
        
        # Pad to 50 dimensions
        while len(obs) < 50:
            obs.append(0.0)
        
        return np.array(obs[:50], dtype=np.float32)
    
    def render(self):
        """Render environment state"""
        if len(self.defense_history) > 0:
            recent = self.defense_history[-10:]
            accuracy = sum(1 for d in recent if d['correct']) / len(recent)
            block_rate = sum(1 for d in recent if d['blocked']) / len(recent)
            
            print(f"\nStep {self.current_step}/{self.max_steps}")
            print(f"Recent Accuracy: {accuracy:.2%}")
            print(f"Recent Block Rate: {block_rate:.2%}")
            print(f"Episode Stats: {self.episode_stats}")


# Test environment
if __name__ == "__main__":
    import sys
    from pathlib import Path
    # Add project root to path
    project_root = Path(__file__).parent.parent
    sys.path.insert(0, str(project_root))
    
    from data.sqli_dataset import SQLInjectionDataset
    
    # Create dataset
    dataset_loader = SQLInjectionDataset(n_attacks=1000, n_benign=1000)
    df = dataset_loader.generate_dataset()
    
    # Create environment
    env = SQLInjectionEnv(dataset=df, dataset_loader=dataset_loader, max_steps=100)
    
    # Test episode
    obs, info = env.reset()
    print("Initial observations:")
    print(f"Red obs shape: {obs['red'].shape}")
    print(f"Blue obs shape: {obs['blue'].shape}")
    
    # Random actions
    for step in range(10):
        actions = {
            'red': env.red_action_space.sample(),
            'blue': env.blue_action_space.sample()
        }
        
        obs, rewards, dones, truncs, info = env.step(actions)
        
        print(f"\nStep {step+1}:")
        print(f"  Red reward: {rewards['red']:.1f}, Blue reward: {rewards['blue']:.1f}")
        print(f"  Attack: {info['is_attack']}, Success: {info['attack_successful']}")
    
    print("\n✅ Environment test complete!")
