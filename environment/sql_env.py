"""
SQL Injection RL Environment - ENHANCED VERSION
✅ Richer reward shaping with multi-objective signals
✅ Attack diversity tracking
✅ Better state representation
✅ Support for Tier 2 PPO training
✅ Comprehensive info dict for analysis
"""

import gymnasium as gym
from gymnasium import spaces
import numpy as np
import sys
from pathlib import Path
from collections import deque

# Add project root to path for imports
_project_root = Path(__file__).parent.parent
if str(_project_root) not in sys.path:
    sys.path.insert(0, str(_project_root))

class SQLInjectionEnv(gym.Env):
    """
    Multi-agent environment for SQL injection attack/defense - ENHANCED
    
    Agents:
    - Red: Generates SQL injection attacks
    - Blue: Detects and responds to attacks (hierarchical)
    
    State: Query features (127-dim) + defense history
    Actions: Red=attack type (30), Blue=response decision (4)
    Rewards: Enhanced reward shaping with multiple signals
    
    IMPROVEMENTS:
    - Richer reward signals (diversity, novelty, cost)
    - Attack type tracking for diversity rewards
    - Better history tracking for context
    - Support for PPO training
    - Comprehensive metrics
    """
    
    metadata = {'render_modes': ['human']}
    
    def __init__(self, dataset, dataset_loader=None, max_steps=1000):
        super().__init__()
        
        self.dataset = dataset
        self.dataset_loader = dataset_loader
        self.max_steps = max_steps
        self.current_step = 0
        
        # Action spaces (EXPANDED for red)
        self.red_action_space = spaces.Discrete(30)  # 30 attack variants (up from 20)
        self.blue_action_space = spaces.Discrete(4)  # Allow, Block, Sanitize, Throttle
        
        # Observation spaces
        self.red_observation_space = spaces.Box(
            low=0, high=1, shape=(50,), dtype=np.float32
        )
        
        self.blue_observation_space = spaces.Box(
            low=0, high=10, shape=(127,), dtype=np.float32
        )
        
        # Enhanced tracking
        self.attack_type_history = deque(maxlen=100)  # Track attack diversity
        self.red_action_counts = np.zeros(30)  # Count each attack type
        self.consecutive_same_action = 0
        self.last_red_action = None
        
        # Performance tracking
        self.cumulative_metrics = {
            'total_attacks': 0,
            'successful_attacks': 0,
            'blocked_attacks': 0,
            'false_positives': 0,
            'true_negatives': 0,
            'novel_attacks': 0,
            'repeated_attacks': 0,
        }
        
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
            'true_negatives': 0,
            'total_cost': 0.0,  # Track computational cost
        }
        
        # Reset attack diversity tracking (keep across episodes for learning)
        self.consecutive_same_action = 0
        self.last_red_action = None
        
        # Shuffle dataset for new episode
        self.dataset_indices = np.random.permutation(len(self.dataset))
        self.current_idx = 0
        
        # Get first observation
        obs = self._get_observations()
        info = self._get_info_dict()
        
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
        
        # Track red action diversity
        self._track_red_action(red_action)
        
        # Get current query
        query_idx = self.dataset_indices[self.current_idx]
        query_data = self.dataset.iloc[query_idx]
        
        is_attack = query_data['label'] == 1
        
        # Blue decision
        # 0=Allow, 1=Block, 2=Sanitize, 3=Throttle
        blue_blocks = blue_action in [1, 2, 3]
        
        # Determine outcome
        attack_successful = is_attack and not blue_blocks
        false_positive = not is_attack and blue_blocks
        true_positive = is_attack and blue_blocks
        true_negative = not is_attack and not blue_blocks
        
        # Calculate ENHANCED rewards
        red_reward = self._calculate_red_reward_enhanced(
            attack_successful, blue_blocks, red_action, is_attack
        )
        blue_reward = self._calculate_blue_reward_enhanced(
            true_positive, false_positive, true_negative, blue_action, is_attack
        )
        
        # Track computational cost
        action_costs = {0: 0.1, 1: 0.5, 2: 1.0, 3: 0.8}  # Allow, Block, Sanitize, Throttle
        self.episode_stats['total_cost'] += action_costs.get(blue_action, 0.5)
        
        # Update statistics
        if attack_successful:
            self.episode_stats['attacks_successful'] += 1
            self.cumulative_metrics['successful_attacks'] += 1
        if true_positive:
            self.episode_stats['attacks_blocked'] += 1
            self.cumulative_metrics['blocked_attacks'] += 1
        if false_positive:
            self.episode_stats['false_positives'] += 1
            self.cumulative_metrics['false_positives'] += 1
        if true_negative:
            self.episode_stats['true_negatives'] += 1
            self.cumulative_metrics['true_negatives'] += 1
        
        if is_attack:
            self.cumulative_metrics['total_attacks'] += 1
        
        # Update history
        self.attack_history.append({
            'action': red_action,
            'success': attack_successful,
            'type': query_data.get('attack_type', 'unknown'),
            'is_novel': self._is_novel_attack(red_action),
        })
        
        self.defense_history.append({
            'action': blue_action,
            'blocked': blue_blocks,
            'correct': true_positive or true_negative,
            'confidence': np.random.uniform(0.5, 1.0),  # Placeholder
            'cost': action_costs.get(blue_action, 0.5),
        })
        
        # Move to next step
        self.current_step += 1
        self.current_idx = (self.current_idx + 1) % len(self.dataset)
        
        # Check if episode is done
        terminated = self.current_step >= self.max_steps
        truncated = False
        
        # Get next observations
        obs = self._get_observations() if not terminated else self._get_zero_observations()
        
        # Enhanced info dict
        info = self._get_info_dict(
            attack_successful=attack_successful,
            false_positive=false_positive,
            true_positive=true_positive,
            is_attack=is_attack,
            query_type=query_data.get('attack_type', 'unknown'),
            red_action=red_action,
            blue_action=blue_action,
        )
        
        # Package for multi-agent
        observations = {'red': obs['red'], 'blue': obs['blue']}
        rewards = {'red': red_reward, 'blue': blue_reward}
        terminateds = {'red': terminated, 'blue': terminated}
        truncateds = {'red': truncated, 'blue': truncated}
        
        return observations, rewards, terminateds, truncateds, info
    
    def _track_red_action(self, action):
        """Track red action for diversity metrics"""
        self.red_action_counts[action] += 1
        
        if action == self.last_red_action:
            self.consecutive_same_action += 1
        else:
            self.consecutive_same_action = 1
        
        self.last_red_action = action
        self.attack_type_history.append(action)
    
    def _is_novel_attack(self, action):
        """Check if attack type is relatively novel (used <5% of time)"""
        total = self.red_action_counts.sum()
        if total < 10:
            return True  # Early in training, everything is novel
        
        usage_rate = self.red_action_counts[action] / total
        return usage_rate < 0.05
    
    def _calculate_red_reward_enhanced(self, success, blocked, action, is_attack):
        """
        Enhanced red agent reward with multiple signals
        
        Components:
        1. Success reward: +10 for bypass
        2. Block penalty: -2 for detection
        3. Diversity bonus: +2 for novel attack types
        4. Exploration penalty: -1 for repeating same action >5 times
        5. False alarm bonus: +3 for causing false positive (wasting blue resources)
        """
        reward = 0.0
        
        # 1. Success reward
        if success:
            reward += 10.0
        
        # 2. Block penalty
        if blocked:
            reward -= 2.0
        
        # 3. Diversity bonus (encourage exploring different attack types)
        if self._is_novel_attack(action):
            reward += 2.0
            self.cumulative_metrics['novel_attacks'] += 1
        
        # 4. Repetition penalty (discourage getting stuck in local optima)
        if self.consecutive_same_action > 5:
            reward -= 1.0
            self.cumulative_metrics['repeated_attacks'] += 1
        
        # 5. Resource exhaustion bonus (if blue uses expensive action on benign)
        if not is_attack and blocked:
            reward += 3.0  # Red benefits from blue wasting resources
        
        # NaN protection
        if not isinstance(reward, (int, float)) or np.isnan(reward) or np.isinf(reward):
            reward = 0.0
        
        return float(reward)
    
    def _calculate_blue_reward_enhanced(self, tp, fp, tn, action, is_attack):
        """
        Enhanced blue agent reward with multiple objectives
        
        Components:
        1. Detection reward: +5 for blocking attacks
        2. False positive penalty: -3 (user friction)
        3. True negative reward: +1 (correct allow)
        4. Action cost: -0.5 to -1.0 depending on action
        5. Efficiency bonus: +1 for low-cost correct decisions
        """
        reward = 0.0
        
        # 1. Detection reward
        if tp:
            reward += 5.0
        
        # 2. False positive penalty (important for UX)
        if fp:
            reward -= 3.0
        
        # 3. True negative reward
        if tn:
            reward += 1.0
        
        # 4. Action costs (encourage efficient responses)
        action_costs = {
            0: 0.0,   # Allow - no cost
            1: -0.3,  # Block - small cost
            2: -0.5,  # Sanitize - processing cost
            3: -1.0,  # Throttle - high cost (impacts all users)
        }
        reward += action_costs.get(action, 0.0)
        
        # 5. Efficiency bonus (correct decision with low-cost action)
        if (tp or tn) and action in [0, 1]:  # Allow or simple block
            reward += 0.5
        
        # 6. Penalty for expensive mistakes
        if (fp or not tp) and action == 3:  # Wrong throttle decision
            reward -= 1.5  # Extra penalty for expensive wrong action
        
        # NaN protection
        if not isinstance(reward, (int, float)) or np.isnan(reward) or np.isinf(reward):
            reward = 0.0
        
        return float(reward)
    
    def _get_observations(self):
        """Get current observations for both agents"""
        # Get current query
        query_idx = self.dataset_indices[self.current_idx]
        query_data = self.dataset.iloc[query_idx]
        
        # Blue observes query features
        if 'features' in query_data and query_data['features'] is not None:
            blue_obs = np.asarray(query_data['features'], dtype=np.float32)
        elif self.dataset_loader is not None:
            blue_obs = self.dataset_loader.extract_features(query_data['query'])
        else:
            from data.sqli_dataset import SQLInjectionDataset
            loader = SQLInjectionDataset()
            blue_obs = loader.extract_features(query_data['query'])
        
        # Red observes defense patterns (ENHANCED)
        red_obs = self._get_red_observation_enhanced()
        
        return {'red': red_obs, 'blue': blue_obs}
    
    def _get_zero_observations(self):
        """Return zero observations for terminal state"""
        return {
            'red': np.zeros(50, dtype=np.float32),
            'blue': np.zeros(127, dtype=np.float32)
        }
    
    def _get_red_observation_enhanced(self):
        """
        Enhanced red agent observation with more context
        
        50-dim vector:
        - [0]: Block rate (last 20)
        - [1]: Accuracy rate (last 20)
        - [2]: Average confidence (last 20)
        - [3-6]: Action distribution (4 actions)
        - [7]: Attack success rate (last 20)
        - [8]: Diversity of blue actions (entropy)
        - [9]: Average cost of blue actions
        - [10-39]: Attack type success rates (30 attack types)
        - [40-49]: Reserved/padding
        """
        obs = []
        
        if len(self.defense_history) == 0:
            return np.zeros(50, dtype=np.float32)
        
        recent = self.defense_history[-20:]
        
        # 0. Block rate
        block_rate = sum(1 for d in recent if d['blocked']) / len(recent)
        obs.append(block_rate)
        
        # 1. Accuracy rate
        accuracy = sum(1 for d in recent if d['correct']) / len(recent)
        obs.append(accuracy)
        
        # 2. Average confidence
        avg_confidence = np.mean([d['confidence'] for d in recent])
        obs.append(avg_confidence)
        
        # 3-6. Action distribution
        for action_id in range(4):
            action_freq = sum(1 for d in recent if d['action'] == action_id) / len(recent)
            obs.append(action_freq)
        
        # 7. Attack success rate
        if len(self.attack_history) > 0:
            recent_attacks = self.attack_history[-20:]
            success_rate = sum(1 for a in recent_attacks if a['success']) / len(recent_attacks)
            obs.append(success_rate)
        else:
            obs.append(0.0)
        
        # 8. Diversity of blue actions (entropy)
        action_counts = [sum(1 for d in recent if d['action'] == i) for i in range(4)]
        action_probs = np.array(action_counts) / max(sum(action_counts), 1)
        entropy = -np.sum(action_probs * np.log(action_probs + 1e-8))
        obs.append(entropy / np.log(4))  # Normalize by max entropy
        
        # 9. Average cost
        avg_cost = np.mean([d['cost'] for d in recent])
        obs.append(avg_cost)
        
        # 10-39. Attack type success rates (30 types)
        for attack_type in range(30):
            type_attacks = [a for a in self.attack_history[-50:] if a['action'] == attack_type]
            if len(type_attacks) > 0:
                type_success = sum(1 for a in type_attacks if a['success']) / len(type_attacks)
                obs.append(type_success)
            else:
                obs.append(0.0)
        
        # 40-49. Padding
        while len(obs) < 50:
            obs.append(0.0)
        
        return np.array(obs[:50], dtype=np.float32)
    
    def _get_info_dict(self, **kwargs):
        """
        Create comprehensive info dictionary
        
        Returns:
            dict with episode statistics and metrics
        """
        info = {
            'episode_stats': self.episode_stats.copy(),
            'current_step': self.current_step,
            'max_steps': self.max_steps,
        }
        
        # Add any additional kwargs
        info.update(kwargs)
        
        # Add diversity metrics if we have attack history
        if len(self.attack_history) > 0:
            unique_attacks = len(set(a['action'] for a in self.attack_history[-50:]))
            info['attack_diversity'] = unique_attacks / 30  # Normalize by total types
            info['novel_attack_rate'] = sum(1 for a in self.attack_history[-20:] if a.get('is_novel', False)) / min(20, len(self.attack_history))
        
        # Add blue performance metrics
        if len(self.defense_history) > 0:
            info['avg_blue_cost'] = np.mean([d['cost'] for d in self.defense_history[-20:]])
            info['blue_accuracy'] = sum(1 for d in self.defense_history[-20:] if d['correct']) / min(20, len(self.defense_history))
        
        # Add cumulative metrics
        info['cumulative_metrics'] = self.cumulative_metrics.copy()
        
        return info
    
    def get_attack_distribution(self):
        """
        Get distribution of attack types used
        Useful for analyzing red agent behavior
        """
        total = self.red_action_counts.sum()
        if total == 0:
            return np.zeros(30)
        
        return self.red_action_counts / total
    
    def render(self):
        """Render environment state"""
        if len(self.defense_history) > 0:
            recent = self.defense_history[-10:]
            accuracy = sum(1 for d in recent if d['correct']) / len(recent)
            block_rate = sum(1 for d in recent if d['blocked']) / len(recent)
            avg_cost = np.mean([d['cost'] for d in recent])
            
            print(f"\nStep {self.current_step}/{self.max_steps}")
            print(f"Recent Accuracy: {accuracy:.2%}")
            print(f"Recent Block Rate: {block_rate:.2%}")
            print(f"Average Cost: {avg_cost:.2f}")
            print(f"Episode Stats: {self.episode_stats}")
            
            if len(self.attack_history) > 0:
                recent_attacks = self.attack_history[-10:]
                success_rate = sum(1 for a in recent_attacks if a['success']) / len(recent_attacks)
                unique = len(set(a['action'] for a in recent_attacks))
                print(f"Red Success Rate: {success_rate:.2%}")
                print(f"Attack Diversity: {unique}/10 unique types")


# Test environment
if __name__ == "__main__":
    import sys
    from pathlib import Path
    project_root = Path(__file__).parent.parent
    sys.path.insert(0, str(project_root))
    
    from data.sqli_dataset import SQLInjectionDataset
    
    # Create dataset
    dataset_loader = SQLInjectionDataset(n_attacks=1000, n_benign=1000)
    df = dataset_loader.generate_dataset()
    
    # Precompute features
    print("Precomputing features...")
    df['features'] = [dataset_loader.extract_features(q) for q in df['query']]
    
    # Create environment
    env = SQLInjectionEnv(dataset=df, dataset_loader=dataset_loader, max_steps=100)
    
    # Test episode
    obs, info = env.reset()
    print("Initial observations:")
    print(f"Red obs shape: {obs['red'].shape}")
    print(f"Blue obs shape: {obs['blue'].shape}")
    print(f"Initial info keys: {info.keys()}")
    
    # Random actions
    total_red_reward = 0
    total_blue_reward = 0
    
    for step in range(20):
        actions = {
            'red': env.red_action_space.sample(),
            'blue': env.blue_action_space.sample()
        }
        
        obs, rewards, dones, truncs, info = env.step(actions)
        
        total_red_reward += rewards['red']
        total_blue_reward += rewards['blue']
        
        if step % 5 == 0:
            print(f"\nStep {step+1}:")
            print(f"  Red reward: {rewards['red']:.1f}, Blue reward: {rewards['blue']:.1f}")
            print(f"  Attack: {info['is_attack']}, Success: {info['attack_successful']}")
            if 'attack_diversity' in info:
                print(f"  Attack diversity: {info['attack_diversity']:.2%}")
    
    print(f"\n=== Episode Summary ===")
    print(f"Total Red Reward: {total_red_reward:.1f}")
    print(f"Total Blue Reward: {total_blue_reward:.1f}")
    print(f"Final episode stats: {info['episode_stats']}")
    print(f"Cumulative metrics: {info['cumulative_metrics']}")
    
    # Test attack distribution
    attack_dist = env.get_attack_distribution()
    print(f"\nAttack distribution (top 5):")
    top_5 = np.argsort(attack_dist)[-5:][::-1]
    for idx in top_5:
        print(f"  Attack {idx}: {attack_dist[idx]:.2%}")
    
    print("\n✅ Environment test complete!")