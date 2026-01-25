"""
Adversarial Training Pipeline
Red-Blue co-evolutionary training
Implements Nash equilibrium convergence through competitive learning
"""

import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from tqdm import tqdm
import os
import json
from datetime import datetime
import torch

# Import our components
import sys
sys.path.append('..')
from data.sqli_dataset import SQLInjectionDataset
from environment.sql_env import SQLInjectionEnv
from agents.blue_tier1 import Tier1Agent
from agents.red_agent import RedAgent


class AdversarialTrainer:
    """
    Manages adversarial training between red and blue agents
    Tracks metrics, saves checkpoints, generates visualizations
    """
    
    def __init__(self, 
                 n_episodes=5000,
                 max_steps_per_episode=1000,
                 save_freq=500,
                 eval_freq=100,
                 save_dir='results',
                 device=None,
                 use_gpu=True):
        
        self.n_episodes = n_episodes
        self.max_steps_per_episode = max_steps_per_episode
        self.save_freq = save_freq
        self.eval_freq = eval_freq
        
        # GPU setup
        if device is None:
            if use_gpu and torch.cuda.is_available():
                self.device = torch.device('cuda')
                print(f"🚀 Using GPU: {torch.cuda.get_device_name(0)}")
                print(f"   CUDA Version: {torch.version.cuda}")
                print(f"   GPU Memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.2f} GB")
            else:
                self.device = torch.device('cpu')
                if use_gpu:
                    print("⚠️  GPU requested but not available. Using CPU.")
                else:
                    print("💻 Using CPU")
        else:
            self.device = device
            print(f"Using specified device: {self.device}")
        
        # Create save directories (use absolute path)
        save_dir = os.path.abspath(save_dir)
        self.save_dir = save_dir
        os.makedirs(f"{save_dir}/models", exist_ok=True)
        os.makedirs(f"{save_dir}/plots", exist_ok=True)
        os.makedirs(f"{save_dir}/logs", exist_ok=True)
        print(f"📁 Save directory: {save_dir}")
        
        # Metrics tracking
        self.metrics = {
            'episode_rewards_red': [],
            'episode_rewards_blue': [],
            'attack_success_rate': [],
            'detection_accuracy': [],
            'false_positive_rate': [],
            'blue_win_rate': [],
            'red_epsilon': [],
            'episode_lengths': []
        }
        
        # Dataset and environment
        print("Creating dataset...")
        dataset_loader = SQLInjectionDataset(n_attacks=5000, n_benign=5000)
        self.dataset = dataset_loader.generate_dataset()
        self.dataset_loader = dataset_loader
        
        print("Creating environment...")
        self.env = SQLInjectionEnv(
            dataset=self.dataset,
            dataset_loader=self.dataset_loader,
            max_steps=max_steps_per_episode
        )
        
        # Agents - ensure both use the same GPU device
        print("Initializing agents on GPU...")
        self.blue_agent = Tier1Agent(learning_rate=3e-4, device=self.device)
        self.red_agent = RedAgent(learning_rate=1e-4, device=self.device)
        
        print(f"✅ Blue agent device: {self.blue_agent.device}")
        print(f"✅ Red agent device: {self.red_agent.device}")
        
        # Enable cuDNN benchmarking for faster training (if GPU available)
        if self.device.type == 'cuda':
            torch.backends.cudnn.benchmark = True
            torch.backends.cudnn.deterministic = False
            print("✅ cuDNN benchmarking enabled for faster training")
            
            # Clear GPU cache
            torch.cuda.empty_cache()
            print(f"✅ GPU memory cleared. Available: {torch.cuda.get_device_properties(0).total_memory / 1e9:.2f} GB")
        
    def train(self):
        """Main training loop"""
        print(f"\n{'='*60}")
        print(f"Starting Adversarial Training: {self.n_episodes} episodes")
        print(f"{'='*60}\n")
        
        # GPU memory management
        if self.device.type == 'cuda':
            torch.cuda.empty_cache()
        
        for episode in tqdm(range(self.n_episodes), desc="Training"):
            # Run episode
            episode_metrics = self._run_episode(episode)
            
            # Store metrics
            self._update_metrics(episode_metrics)
            
            # Evaluation
            if (episode + 1) % self.eval_freq == 0:
                self._evaluate(episode + 1)
            
            # Save checkpoint
            if (episode + 1) % self.save_freq == 0:
                self._save_checkpoint(episode + 1)
                # Clear GPU cache periodically
                if self.device.type == 'cuda' and (episode + 1) % (self.save_freq * 2) == 0:
                    torch.cuda.empty_cache()
        
        # Final save and plots
        self._save_checkpoint(self.n_episodes, final=True)
        self._generate_plots()
        self._save_metrics()
        
        # Final GPU memory report
        if self.device.type == 'cuda':
            print(f"\nGPU Memory Usage:")
            print(f"  Allocated: {torch.cuda.memory_allocated() / 1e9:.2f} GB")
            print(f"  Cached: {torch.cuda.memory_reserved() / 1e9:.2f} GB")
        
        print(f"\n{'='*60}")
        print("Training Complete!")
        print(f"{'='*60}\n")
    
    def _run_episode(self, episode_num):
        """Run single episode of adversarial interaction"""
        obs, _ = self.env.reset()
        
        episode_reward_red = 0
        episode_reward_blue = 0
        episode_stats = {
            'attacks_successful': 0,
            'attacks_blocked': 0,
            'false_positives': 0,
            'true_negatives': 0,
            'total_attacks': 0,
            'total_benign': 0
        }
        
        for step in range(self.max_steps_per_episode):
            # Agent actions
            blue_action, _ = self.blue_agent.select_action(
                obs['blue'], 
                deterministic=False
            )
            red_action = self.red_agent.select_action(
                obs['red'], 
                deterministic=False
            )
            
            # Environment step
            actions = {'red': red_action, 'blue': blue_action}
            next_obs, rewards, dones, truncs, info = self.env.step(actions)
            
            # Store transitions
            self.red_agent.store_transition(
                obs['red'], red_action, rewards['red'], 
                next_obs['red'], dones['red']
            )
            
            # Track stats
            episode_reward_red += rewards['red']
            episode_reward_blue += rewards['blue']
            
            if info['is_attack']:
                episode_stats['total_attacks'] += 1
                if info['attack_successful']:
                    episode_stats['attacks_successful'] += 1
                else:
                    episode_stats['attacks_blocked'] += 1
            else:
                episode_stats['total_benign'] += 1
                if info['false_positive']:
                    episode_stats['false_positives'] += 1
                else:
                    episode_stats['true_negatives'] += 1
            
            # Train agents
            if step % 4 == 0:  # Train every 4 steps
                # Red agent training (DQN)
                red_loss = self.red_agent.train_step()
                
                # Blue agent training (simplified supervised learning)
                # In full version, this would be more sophisticated
                if len(self.env.defense_history) >= 32:
                    self._train_blue_agent()
            
            # Record red agent success for novelty tracking
            self.red_agent.record_attack_result(info['attack_successful'])
            
            obs = next_obs
            
            if dones['red'] or dones['blue']:
                break
        
        # Calculate episode metrics
        episode_metrics = {
            'reward_red': episode_reward_red,
            'reward_blue': episode_reward_blue,
            'attack_success_rate': episode_stats['attacks_successful'] / max(episode_stats['total_attacks'], 1),
            'detection_accuracy': episode_stats['attacks_blocked'] / max(episode_stats['total_attacks'], 1),
            'false_positive_rate': episode_stats['false_positives'] / max(episode_stats['total_benign'], 1),
            'blue_win_rate': episode_stats['attacks_blocked'] / max(episode_stats['total_attacks'], 1),
            'red_epsilon': self.red_agent.epsilon,
            'episode_length': step + 1
        }
        
        return episode_metrics
    
    def _train_blue_agent(self):
        """Train blue agent on recent experiences (GPU-optimized)"""
        # Get recent defense history
        recent_history = self.env.defense_history[-32:]
        
        # Extract observations and ground truth
        observations = []
        labels = []
        
        for i, history in enumerate(recent_history):
            # Get corresponding query index
            idx = max(0, self.env.current_idx - len(recent_history) + i)
            query_idx = self.env.dataset_indices[idx % len(self.env.dataset)]
            query_data = self.dataset.iloc[query_idx]
            
            # Extract features
            obs = self.dataset_loader.extract_features(query_data['query'])
            observations.append(obs)
            
            # Label: 0=Pass, 1=Escalate, 2=Suggest, 3=Block
            # Simple heuristic: if attack -> block, else -> pass
            if query_data['label'] == 1:
                labels.append(3)  # Block
            else:
                labels.append(0)  # Pass
        
        # Convert to numpy arrays (CPU) then move to GPU in train_step
        observations = np.array(observations, dtype=np.float32)
        labels = np.array(labels, dtype=np.int64)
        
        # Train (tensors will be moved to GPU inside train_step)
        loss, acc = self.blue_agent.train_step(observations, labels)
    
    def _update_metrics(self, episode_metrics):
        """Update running metrics"""
        self.metrics['episode_rewards_red'].append(episode_metrics['reward_red'])
        self.metrics['episode_rewards_blue'].append(episode_metrics['reward_blue'])
        self.metrics['attack_success_rate'].append(episode_metrics['attack_success_rate'])
        self.metrics['detection_accuracy'].append(episode_metrics['detection_accuracy'])
        self.metrics['false_positive_rate'].append(episode_metrics['false_positive_rate'])
        self.metrics['blue_win_rate'].append(episode_metrics['blue_win_rate'])
        self.metrics['red_epsilon'].append(episode_metrics['red_epsilon'])
        self.metrics['episode_lengths'].append(episode_metrics['episode_length'])
    
    def _evaluate(self, episode):
        """Evaluate current performance"""
        recent = 100
        avg_red_reward = np.mean(self.metrics['episode_rewards_red'][-recent:])
        avg_blue_reward = np.mean(self.metrics['episode_rewards_blue'][-recent:])
        avg_attack_success = np.mean(self.metrics['attack_success_rate'][-recent:])
        avg_detection = np.mean(self.metrics['detection_accuracy'][-recent:])
        avg_fp = np.mean(self.metrics['false_positive_rate'][-recent:])
        
        print(f"\n--- Episode {episode} Evaluation ---")
        print(f"Red Reward: {avg_red_reward:.2f}")
        print(f"Blue Reward: {avg_blue_reward:.2f}")
        print(f"Attack Success Rate: {avg_attack_success:.2%}")
        print(f"Detection Accuracy: {avg_detection:.2%}")
        print(f"False Positive Rate: {avg_fp:.2%}")
        print(f"Red Epsilon: {self.red_agent.epsilon:.3f}")
        
        # GPU memory info
        if self.device.type == 'cuda':
            allocated = torch.cuda.memory_allocated() / 1e9
            cached = torch.cuda.memory_reserved() / 1e9
            print(f"GPU Memory: {allocated:.2f} GB allocated, {cached:.2f} GB cached")
        print()
    
    def _save_checkpoint(self, episode, final=False):
        """Save model checkpoints"""
        suffix = 'final' if final else f'ep{episode}'
        
        try:
            blue_path = os.path.join(self.save_dir, "models", f"blue_agent_{suffix}.pt")
            red_path = os.path.join(self.save_dir, "models", f"red_agent_{suffix}.pt")
            
            self.blue_agent.save(blue_path)
            self.red_agent.save(red_path)
            
            # Verify files were created
            if os.path.exists(blue_path) and os.path.exists(red_path):
                print(f"✅ Checkpoint saved: {suffix}")
                print(f"   Blue: {blue_path}")
                print(f"   Red: {red_path}")
            else:
                print(f"⚠️  Warning: Checkpoint files may not have been created")
        except Exception as e:
            print(f"❌ Error saving checkpoint: {e}")
    
    def _save_metrics(self):
        """Save metrics to JSON"""
        filepath = os.path.join(self.save_dir, "logs", "training_metrics.json")
        
        try:
            # Convert to serializable format
            metrics_serializable = {
                k: [float(v) for v in values] 
                for k, values in self.metrics.items()
            }
            
            with open(filepath, 'w') as f:
                json.dump(metrics_serializable, f, indent=2)
            
            if os.path.exists(filepath):
                print(f"✅ Metrics saved to {filepath}")
            else:
                print(f"⚠️  Warning: Metrics file may not have been created")
        except Exception as e:
            print(f"❌ Error saving metrics: {e}")
    
    def _generate_plots(self):
        """Generate training visualization plots"""
        sns.set_style("whitegrid")
        
        # Create figure with subplots
        fig, axes = plt.subplots(3, 2, figsize=(15, 12))
        fig.suptitle('Adversarial Training Results', fontsize=16, fontweight='bold')
        
        # Plot 1: Rewards
        ax = axes[0, 0]
        episodes = range(len(self.metrics['episode_rewards_red']))
        ax.plot(episodes, self._smooth(self.metrics['episode_rewards_red']), 
                label='Red Agent', color='red', alpha=0.8)
        ax.plot(episodes, self._smooth(self.metrics['episode_rewards_blue']), 
                label='Blue Agent', color='blue', alpha=0.8)
        ax.set_xlabel('Episode')
        ax.set_ylabel('Reward')
        ax.set_title('Episode Rewards')
        ax.legend()
        ax.grid(True, alpha=0.3)
        
        # Plot 2: Attack Success Rate
        ax = axes[0, 1]
        ax.plot(episodes, self._smooth(self.metrics['attack_success_rate']), 
                color='darkred', linewidth=2)
        ax.set_xlabel('Episode')
        ax.set_ylabel('Success Rate')
        ax.set_title('Attack Success Rate (Red Performance)')
        ax.grid(True, alpha=0.3)
        
        # Plot 3: Detection Accuracy
        ax = axes[1, 0]
        ax.plot(episodes, self._smooth(self.metrics['detection_accuracy']), 
                color='darkblue', linewidth=2)
        ax.set_xlabel('Episode')
        ax.set_ylabel('Accuracy')
        ax.set_title('Detection Accuracy (Blue Performance)')
        ax.grid(True, alpha=0.3)
        
        # Plot 4: False Positive Rate
        ax = axes[1, 1]
        ax.plot(episodes, self._smooth(self.metrics['false_positive_rate']), 
                color='orange', linewidth=2)
        ax.set_xlabel('Episode')
        ax.set_ylabel('FP Rate')
        ax.set_title('False Positive Rate')
        ax.grid(True, alpha=0.3)
        
        # Plot 5: Blue Win Rate
        ax = axes[2, 0]
        ax.plot(episodes, self._smooth(self.metrics['blue_win_rate']), 
                color='green', linewidth=2)
        ax.set_xlabel('Episode')
        ax.set_ylabel('Win Rate')
        ax.set_title('Blue Win Rate (Attacks Blocked)')
        ax.grid(True, alpha=0.3)
        
        # Plot 6: Red Epsilon (Exploration)
        ax = axes[2, 1]
        ax.plot(episodes, self.metrics['red_epsilon'], 
                color='purple', linewidth=2)
        ax.set_xlabel('Episode')
        ax.set_ylabel('Epsilon')
        ax.set_title('Red Agent Exploration Rate')
        ax.grid(True, alpha=0.3)
        
        plt.tight_layout()
        
        # Save
        filepath = os.path.join(self.save_dir, "plots", "training_results.png")
        try:
            plt.savefig(filepath, dpi=300, bbox_inches='tight')
            if os.path.exists(filepath):
                print(f"✅ Training plots saved to {filepath}")
            else:
                print(f"⚠️  Warning: Plot file may not have been created")
        except Exception as e:
            print(f"❌ Error saving plots: {e}")
        
        plt.close()
    
    def _smooth(self, data, window=50):
        """Smooth data for plotting"""
        if len(data) < window:
            return data
        smoothed = np.convolve(data, np.ones(window)/window, mode='valid')
        # Pad to original length
        pad_length = len(data) - len(smoothed)
        return np.concatenate([data[:pad_length], smoothed])


# Main execution
if __name__ == "__main__":
    # Create trainer
    trainer = AdversarialTrainer(
        n_episodes=5000,
        max_steps_per_episode=1000,
        save_freq=500,
        eval_freq=100
    )
    
    # Run training
    trainer.train()
