"""
Hierarchical Adversarial Training Pipeline - FIXED VERSION
✅ Fixed reward storage bug (reward_red → episode_rewards_red)
✅ Added curriculum learning (progressive difficulty)
✅ Better red-blue balance mechanisms
✅ Improved training dynamics
"""

import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from tqdm import tqdm
import os
import json
from datetime import datetime

# Import components
import sys
sys.path.append('..')
from data.sqli_dataset import SQLInjectionDataset
from environment.sql_env import SQLInjectionEnv
from agents.blue_tier1 import Tier1Agent
from agents.blue_tier2 import Tier2Agent
from agents.master import MasterCoordinator
from agents.red_agent import RedAgent
from evaluation.baselines import ModSecurityBaseline, SimpleMLBaseline


class HierarchicalAdversarialTrainer:
    """
    Hierarchical Multi-Agent RL Training
    Full 3-tier architecture with curriculum learning
    """
    
    def __init__(self, 
                 n_episodes=5000,
                 max_steps_per_episode=1000,
                 save_freq=500,
                 eval_freq=100,
                 save_dir='../results',
                 use_hierarchy=True,
                 curriculum_learning=True):
        
        self.n_episodes = n_episodes
        self.max_steps_per_episode = max_steps_per_episode
        self.save_freq = save_freq
        self.eval_freq = eval_freq
        self.save_dir = save_dir
        self.use_hierarchy = use_hierarchy
        self.curriculum_learning = curriculum_learning
        
        # Create directories
        os.makedirs(f"{save_dir}/models", exist_ok=True)
        os.makedirs(f"{save_dir}/plots", exist_ok=True)
        os.makedirs(f"{save_dir}/logs", exist_ok=True)
        
        # Metrics tracking
        self.metrics = {
            'episode_rewards_red': [],
            'episode_rewards_blue': [],
            'attack_success_rate': [],
            'detection_accuracy': [],
            'false_positive_rate': [],
            'blue_win_rate': [],
            'red_epsilon': [],
            'episode_lengths': [],
            # Hierarchical routing metrics
            'tier1_only_rate': [],
            'tier2_escalated_rate': [],
            'tier3_analyzed_rate': [],
            'immediate_block_rate': [],
        }
        
        # Curriculum tracking
        self.curriculum_phase = 0  # 0=easy, 1=medium, 2=hard
        self.consecutive_perfect_blue = 0
        
        # Dataset
        print("Creating dataset...")
        dataset_loader = SQLInjectionDataset(n_attacks=1000, n_benign=1000)
        self.dataset = dataset_loader.generate_dataset()
        self.dataset_loader = dataset_loader

        # Precompute features once (major speedup)
        print("Precomputing features...")
        self.dataset['features'] = [
            self.dataset_loader.extract_features(q) for q in self.dataset['query']
        ]
        
        print("Creating environment...")
        self.env = SQLInjectionEnv(dataset=self.dataset, dataset_loader=dataset_loader, max_steps=max_steps_per_episode)
        
        # Agents
        print("Initializing hierarchical agents...")
        self.tier1_agent = Tier1Agent(learning_rate=3e-4)
        self.tier2_agent = Tier2Agent(learning_rate=3e-4) if use_hierarchy else None
        self.master = MasterCoordinator() if use_hierarchy else None
        self.red_agent = RedAgent(learning_rate=1e-4)
        
        print(f"Tier 1 device: {self.tier1_agent.device}")
        if self.tier2_agent:
            print(f"Tier 2 device: {self.tier2_agent.device}")
        print(f"Hierarchy enabled: {use_hierarchy}")
        print(f"Curriculum learning: {curriculum_learning}")
        
        # Baselines (for comparison)
        print("Initializing baselines...")
        self.modsec_baseline = ModSecurityBaseline()
        self.ml_baseline = SimpleMLBaseline()
        self._train_ml_baseline()
        
    def _train_ml_baseline(self):
        """Train simple ML baseline on initial dataset"""
        print("Training ML baseline...")
        X = np.asarray(self.dataset['features'].tolist())
        y = self.dataset['label'].values
        
        from sklearn.model_selection import train_test_split
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
        
        self.ml_baseline.train(X_train, y_train)
        metrics = self.ml_baseline.evaluate(X_test, y_test)
        print(f"ML Baseline initial performance: {metrics}")
        
    def train(self):
        """Main hierarchical training loop with curriculum"""
        print(f"\n{'='*60}")
        print(f"Hierarchical Adversarial Training: {self.n_episodes} episodes")
        print(f"Architecture: {'3-Tier Hierarchy' if self.use_hierarchy else 'Tier 1 Only'}")
        print(f"Curriculum: {'Enabled' if self.curriculum_learning else 'Disabled'}")
        print(f"{'='*60}\n")
        
        for episode in tqdm(range(self.n_episodes), desc="Training"):
            # Curriculum adjustment
            if self.curriculum_learning:
                self._adjust_curriculum(episode)
            
            episode_metrics = self._run_hierarchical_episode(episode)
            self._update_metrics(episode_metrics)
            
            # Check for blue dominance
            if episode_metrics['detection_accuracy'] >= 0.99:
                self.consecutive_perfect_blue += 1
            else:
                self.consecutive_perfect_blue = 0
            
            # If blue dominates for 100 episodes, boost red
            if self.consecutive_perfect_blue >= 100:
                print(f"\n⚠️  Blue dominating for 100 episodes - boosting red agent...")
                self._boost_red_agent()
                self.consecutive_perfect_blue = 0

            if (episode + 1) % self.eval_freq == 0:
                self._evaluate(episode + 1)
            
            if (episode + 1) % self.save_freq == 0:
                self._save_checkpoint(episode + 1)
        
        # Final evaluation and comparison
        self._save_checkpoint(self.n_episodes, final=True)
        self._final_baseline_comparison()
        self._generate_plots()
        self._save_metrics()
        
        print(f"\n{'='*60}")
        print("Training Complete!")
        print(f"{'='*60}\n")
    
    def _adjust_curriculum(self, episode):
        """
        Curriculum learning: Start easy, progressively harder
        Phase 0 (ep 0-150): Blue gets 30% less training
        Phase 1 (ep 150-300): Normal training
        Phase 2 (ep 300+): Red gets attack mutation
        """
        if episode == 150:
            self.curriculum_phase = 1
            print("\n📚 Curriculum Phase 1: Normal training begins")
        elif episode == 300:
            self.curriculum_phase = 2
            self.red_agent.enable_mutation = True
            print("\n📚 Curriculum Phase 2: Red attack mutation enabled")
    
    def _boost_red_agent(self):
        """Boost red agent when blue dominates"""
        # Increase epsilon for more exploration
        self.red_agent.epsilon = min(0.6, self.red_agent.epsilon * 1.3)
        print(f"    🔴 Red epsilon boosted to {self.red_agent.epsilon:.3f}")
        
        # Enable mutation if not already
        self.red_agent.enable_mutation = True
        print(f"    🔴 Red mutation enabled")
    
    def _run_hierarchical_episode(self, episode_num):
        """Run episode with hierarchical decision-making"""
        obs, _ = self.env.reset()
        
        # 🔧 FIX: Initialize as float
        episode_reward_red = 0.0
        episode_reward_blue = 0.0
        
        episode_stats = {
            'attacks_successful': 0,
            'attacks_blocked': 0,
            'false_positives': 0,
            'true_negatives': 0,
            'total_attacks': 0,
            'total_benign': 0,
            # Routing stats
            'tier1_only': 0,
            'tier2': 0,
            'tier3': 0,
            'immediate_block': 0,
        }
        
        for step in range(self.max_steps_per_episode):
            # === HIERARCHICAL DECISION FLOW ===
            
            # 1. Tier 1 Detection
            tier1_action, tier1_confidence = self.tier1_agent.select_action(
                obs['blue'], deterministic=False
            )
            
            final_action = tier1_action
            route = 'tier1_only'
            
            # 2. Master Coordination (if hierarchy enabled)
            if self.use_hierarchy and self.master:
                route, master_action, master_conf = self.master.coordinate(
                    tier1_confidence=tier1_confidence,
                    tier1_action=tier1_action,
                    query_features=obs['blue']
                )
                
                # Track routing
                episode_stats[route] = episode_stats.get(route, 0) + 1
                
                # 3. Tier 2 Decision (if escalated)
                if route == 'tier2':
                    context = self.master.get_context_for_tier2()
                    tier1_probs = np.zeros(4)
                    tier1_probs[tier1_action] = tier1_confidence
                    
                    # FIXED: Unpack 4 values from Tier 2
                    tier2_action, tier2_probs, tier2_value, tier2_log_prob = self.tier2_agent.select_action(
                        obs['blue'], tier1_probs, context
                    )
                    final_action = tier2_action
                elif route == 'immediate_block':
                    final_action = 3  # Block
                elif route == 'tier3':
                    final_action = tier1_action
                else:  # tier1_only
                    final_action = tier1_action
            
            # Red agent action - ADAPTIVE with mutation
            blue_blocking_patterns = {
                'recent_block_rate': sum(1 for d in self.env.defense_history[-20:] if d.get('blocked', False)) / max(len(self.env.defense_history[-20:]), 1) if len(self.env.defense_history) > 0 else 0.0
            }
            
            red_action = self.red_agent.select_action_adaptive(
                obs['red'], 
                blue_blocking_patterns, 
                deterministic=False
            )

            # Adapt red's exploration based on blue's strength (less frequent)
            if step % 100 == 0 and len(self.env.defense_history) >= 20:
                recent_defenses = self.env.defense_history[-100:]
                blue_strength = sum(1 for d in recent_defenses if d.get('correct', False)) / len(recent_defenses)
                self.red_agent.adapt_to_blue_defense(blue_strength)
            
            # Environment step
            actions = {'red': red_action, 'blue': final_action}
            next_obs, rewards, dones, truncs, info = self.env.step(actions)
            
            # NaN protection
            if np.isnan(rewards['red']) or np.isinf(rewards['red']):
                rewards['red'] = 0.0
            if np.isnan(rewards['blue']) or np.isinf(rewards['blue']):
                rewards['blue'] = 0.0

            # Store transitions
            self.red_agent.store_transition(
                obs['red'], red_action, rewards['red'], 
                next_obs['red'], dones['red']
            )
            
            # Update master system state
            if self.master:
                self.master.update_system_state(
                    attack_detected=info['true_positive'],
                    false_positive=info['false_positive']
                )
            
            # 🔧 FIX: Accumulate rewards as float
            episode_reward_red += float(rewards['red'])
            episode_reward_blue += float(rewards['blue'])
            
            # Track stats
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
            
            # Train agents - curriculum adjusted
            if self.curriculum_phase == 0:
                # Phase 0: Train red normally, blue less frequently
                if step % 32 == 0:
                    self.red_agent.train_step()
                if step % 64 == 0 and len(self.env.defense_history) >= 32:
                    self._train_blue_agents()
            else:
                # Phase 1+: Normal training
                if step % 32 == 0:
                    red_loss = self.red_agent.train_step()
                    if len(self.env.defense_history) >= 32:
                        self._train_blue_agents()
            
            self.red_agent.record_attack_result(info['attack_successful'])
            obs = next_obs
            
            if dones['red'] or dones['blue']:
                break
        
        # Calculate metrics with NaN protection
        total_queries = max(episode_stats['total_attacks'] + episode_stats['total_benign'], 1)
        
        attack_success_rate = episode_stats['attacks_successful'] / max(episode_stats['total_attacks'], 1)
        detection_accuracy = episode_stats['attacks_blocked'] / max(episode_stats['total_attacks'], 1)
        false_positive_rate = episode_stats['false_positives'] / max(episode_stats['total_benign'], 1)
        blue_win_rate = episode_stats['attacks_blocked'] / max(episode_stats['total_attacks'], 1)
        
        # NaN protection
        attack_success_rate = 0.0 if np.isnan(attack_success_rate) else float(attack_success_rate)
        detection_accuracy = 0.0 if np.isnan(detection_accuracy) else float(detection_accuracy)
        false_positive_rate = 0.0 if np.isnan(false_positive_rate) else float(false_positive_rate)
        blue_win_rate = 0.0 if np.isnan(blue_win_rate) else float(blue_win_rate)
        
        # 🔧 CRITICAL FIX: Use correct key names!
        episode_metrics = {
            'episode_rewards_red': float(episode_reward_red),   # FIXED: was 'reward_red'
            'episode_rewards_blue': float(episode_reward_blue), # FIXED: was 'reward_blue'
            'attack_success_rate': attack_success_rate,
            'detection_accuracy': detection_accuracy,
            'false_positive_rate': false_positive_rate,
            'blue_win_rate': blue_win_rate,
            'red_epsilon': float(self.red_agent.epsilon),
            'episode_length': step + 1,
        }
        
        # Routing metrics (if hierarchy enabled)
        if self.use_hierarchy and total_queries > 0:
            episode_metrics.update({
                'tier1_only_rate': episode_stats.get('tier1_only', 0) / total_queries,
                'tier2_escalated_rate': episode_stats.get('tier2', 0) / total_queries,
                'tier3_analyzed_rate': episode_stats.get('tier3', 0) / total_queries,
                'immediate_block_rate': episode_stats.get('immediate_block', 0) / total_queries,
            })
        
        return episode_metrics
    
    def _train_blue_agents(self):
        """Optimized batched training"""
        if len(self.env.defense_history) < 32:
            return
        
        recent_history = self.env.defense_history[-32:]
        
        # Preallocate arrays
        observations = np.zeros((len(recent_history), 127), dtype=np.float32)
        labels = np.zeros(len(recent_history), dtype=np.int64)
        
        for i, history in enumerate(recent_history):
            idx = max(0, self.env.current_idx - len(recent_history) + i)
            query_idx = self.env.dataset_indices[idx % len(self.env.dataset)]
            query_data = self.dataset.iloc[query_idx]
            
            # Use precomputed features
            observations[i] = query_data['features']
            labels[i] = 3 if query_data['label'] == 1 else 0
        
        # Single batch training
        loss, acc = self.tier1_agent.train_step(observations, labels)
        
    def _update_metrics(self, episode_metrics):
        """Update running metrics"""
        for key, value in episode_metrics.items():
            if key in self.metrics:
                self.metrics[key].append(value)
    
    def _evaluate(self, episode):
        """Evaluate current performance"""
        recent = min(100, episode)
        
        avg_red_reward = np.mean(self.metrics['episode_rewards_red'][-recent:])
        avg_blue_reward = np.mean(self.metrics['episode_rewards_blue'][-recent:])
        avg_attack_success = np.mean(self.metrics['attack_success_rate'][-recent:])
        avg_detection = np.mean(self.metrics['detection_accuracy'][-recent:])
        avg_fp = np.mean(self.metrics['false_positive_rate'][-recent:])
        
        print(f"\n{'='*50}")
        print(f"Episode {episode} Evaluation")
        print(f"{'='*50}")
        print(f"Red Reward:          {avg_red_reward:>8.2f}")
        print(f"Blue Reward:         {avg_blue_reward:>8.2f}")
        print(f"Attack Success:      {avg_attack_success:>7.1%}")
        print(f"Detection Accuracy:  {avg_detection:>7.1%}")
        print(f"False Positive:      {avg_fp:>7.1%}")
        print(f"Red Epsilon:         {self.red_agent.epsilon:>8.3f}")
        
        if self.curriculum_learning:
            print(f"Curriculum Phase:    {self.curriculum_phase}")
        
        if self.use_hierarchy:
            avg_t1 = np.mean(self.metrics['tier1_only_rate'][-recent:])
            avg_t2 = np.mean(self.metrics['tier2_escalated_rate'][-recent:])
            avg_t3 = np.mean(self.metrics['tier3_analyzed_rate'][-recent:])
            avg_block = np.mean(self.metrics['immediate_block_rate'][-recent:])
            
            print(f"\nHierarchical Routing:")
            print(f"  Tier 1 Only:       {avg_t1:>7.1%}")
            print(f"  Tier 2 Escalated:  {avg_t2:>7.1%}")
            print(f"  Tier 3 Analyzed:   {avg_t3:>7.1%}")
            print(f"  Immediate Block:   {avg_block:>7.1%}")
        
        print(f"{'='*50}\n")
    
    def _final_baseline_comparison(self):
        """Compare against baselines on test set"""
        print("\n" + "="*60)
        print("BASELINE COMPARISON")
        print("="*60)
        
        # Evaluate on test data
        test_size = min(1000, len(self.dataset))
        test_indices = np.random.choice(len(self.dataset), test_size, replace=False)
        
        # Web-HMARL evaluation
        hmarl_correct = 0
        hmarl_fp = 0
        hmarl_fn = 0
        
        # ModSecurity evaluation
        modsec_correct = 0
        modsec_fp = 0
        modsec_fn = 0
        
        # ML Baseline evaluation
        ml_correct = 0
        ml_fp = 0
        ml_fn = 0
        
        for idx in tqdm(test_indices, desc="Evaluating baselines"):
            query_data = self.dataset.iloc[idx]
            query = query_data['query']
            true_label = query_data['label']
            features = query_data['features']
            
            # Web-HMARL prediction
            tier1_action, tier1_conf = self.tier1_agent.select_action(features, deterministic=True)
            hmarl_blocks = tier1_action in [2, 3]
            
            if true_label == 1:  # Attack
                if hmarl_blocks:
                    hmarl_correct += 1
                else:
                    hmarl_fn += 1
            else:  # Benign
                if not hmarl_blocks:
                    hmarl_correct += 1
                else:
                    hmarl_fp += 1
            
            # ModSecurity prediction
            modsec_detected, _ = self.modsec_baseline.detect(query)
            if true_label == 1:
                if modsec_detected:
                    modsec_correct += 1
                else:
                    modsec_fn += 1
            else:
                if not modsec_detected:
                    modsec_correct += 1
                else:
                    modsec_fp += 1
            
            # ML Baseline prediction
            ml_pred = self.ml_baseline.predict(features.reshape(1, -1))[0]
            if true_label == ml_pred:
                ml_correct += 1
            elif true_label == 1:
                ml_fn += 1
            else:
                ml_fp += 1
        
        # Calculate metrics
        results = {
            'Web-HMARL': {
                'accuracy': (hmarl_correct / test_size) * 100,
                'false_positive': (hmarl_fp / test_size) * 100,
                'false_negative': (hmarl_fn / test_size) * 100,
            },
            'ModSecurity CRS': {
                'accuracy': (modsec_correct / test_size) * 100,
                'false_positive': (modsec_fp / test_size) * 100,
                'false_negative': (modsec_fn / test_size) * 100,
            },
            'ML Baseline': {
                'accuracy': (ml_correct / test_size) * 100,
                'false_positive': (ml_fp / test_size) * 100,
                'false_negative': (ml_fn / test_size) * 100,
            }
        }
        
        # Print comparison table
        print(f"\n{'Method':<20} {'Accuracy':<12} {'FP Rate':<12} {'FN Rate':<12}")
        print("-" * 60)
        for method, metrics in results.items():
            print(f"{method:<20} {metrics['accuracy']:>10.1f}% {metrics['false_positive']:>10.1f}% {metrics['false_negative']:>10.1f}%")
        
        # Save comparison
        with open(f"{self.save_dir}/logs/baseline_comparison.json", 'w') as f:
            json.dump(results, f, indent=2)
        
        print(f"\nComparison saved to {self.save_dir}/logs/baseline_comparison.json")
        print("="*60 + "\n")
        
        return results
    
    def _save_checkpoint(self, episode, final=False):
        """Save model checkpoints"""
        suffix = 'final' if final else f'ep{episode}'
        
        self.tier1_agent.save(f"{self.save_dir}/models/tier1_{suffix}.pt")
        if self.tier2_agent:
            self.tier2_agent.save(f"{self.save_dir}/models/tier2_{suffix}.pt")
        self.red_agent.save(f"{self.save_dir}/models/red_agent_{suffix}.pt")
        
        if final:
            print("[OK] Final checkpoint saved")
        else:
            print(f"Checkpoint saved: episode {episode}")
    
    def _save_metrics(self):
        """Save metrics to JSON"""
        filepath = f"{self.save_dir}/logs/training_metrics.json"
        
        metrics_serializable = {
            k: [float(v) if not isinstance(v, (list, dict)) else v for v in values] 
            for k, values in self.metrics.items()
        }
        
        with open(filepath, 'w') as f:
            json.dump(metrics_serializable, f, indent=2)
        
        print(f"Metrics saved to {filepath}")
    
    def _generate_plots(self):
        """Generate comprehensive training visualization"""
        sns.set_style("whitegrid")
        
        if len(self.metrics['episode_rewards_red']) == 0:
            print("[WARN] No metrics to plot yet")
            return
        
        if self.use_hierarchy:
            fig, axes = plt.subplots(4, 2, figsize=(16, 16))
        else:
            fig, axes = plt.subplots(3, 2, figsize=(16, 12))
        
        fig.suptitle(
            'Hierarchical Multi-Agent RL Training Results' if self.use_hierarchy else 'Flat MARL Training Results',
            fontsize=16, fontweight='bold'
        )
        
        n_episodes = len(self.metrics['episode_rewards_red'])
        episodes = np.arange(n_episodes)
        
        # Plot 1: Rewards
        ax = axes[0, 0]
        smoothed_red = self._smooth(self.metrics['episode_rewards_red'])
        smoothed_blue = self._smooth(self.metrics['episode_rewards_blue'])
        
        episodes_plot = np.arange(len(smoothed_red))
        
        ax.plot(episodes_plot, smoothed_red, label='Red Agent', color='red', alpha=0.8, linewidth=2)
        ax.plot(episodes_plot[:len(smoothed_blue)], smoothed_blue, label='Blue Agent', color='blue', alpha=0.8, linewidth=2)
        ax.set_xlabel('Episode', fontsize=11)
        ax.set_ylabel('Reward', fontsize=11)
        ax.set_title('Episode Rewards', fontsize=12, fontweight='bold')
        ax.legend()
        ax.grid(True, alpha=0.3)
        
        # Plot 2: Attack Success Rate
        ax = axes[0, 1]
        smoothed = self._smooth(self.metrics['attack_success_rate'])
        episodes_plot = np.arange(len(smoothed))
        ax.plot(episodes_plot, smoothed, color='darkred', linewidth=2)
        ax.set_xlabel('Episode', fontsize=11)
        ax.set_ylabel('Success Rate', fontsize=11)
        ax.set_title('Attack Success Rate (Red Performance)', fontsize=12, fontweight='bold')
        ax.axhline(y=0.0, color='green', linestyle='--', alpha=0.5, label='Target: 0%')
        ax.legend()
        ax.grid(True, alpha=0.3)
        
        # Plot 3: Detection Accuracy
        ax = axes[1, 0]
        smoothed = self._smooth(self.metrics['detection_accuracy'])
        episodes_plot = np.arange(len(smoothed))
        ax.plot(episodes_plot, smoothed, color='darkblue', linewidth=2)
        ax.set_xlabel('Episode', fontsize=11)
        ax.set_ylabel('Accuracy', fontsize=11)
        ax.set_title('Detection Accuracy (Blue Performance)', fontsize=12, fontweight='bold')
        ax.axhline(y=1.0, color='green', linestyle='--', alpha=0.5, label='Target: 100%')
        ax.legend()
        ax.grid(True, alpha=0.3)
        
        # Plot 4: False Positive Rate
        ax = axes[1, 1]
        smoothed = self._smooth(self.metrics['false_positive_rate'])
        episodes_plot = np.arange(len(smoothed))
        ax.plot(episodes_plot, smoothed, color='orange', linewidth=2)
        ax.set_xlabel('Episode', fontsize=11)
        ax.set_ylabel('FP Rate', fontsize=11)
        ax.set_title('False Positive Rate', fontsize=12, fontweight='bold')
        ax.axhline(y=0.0, color='green', linestyle='--', alpha=0.5, label='Target: 0%')
        ax.legend()
        ax.grid(True, alpha=0.3)
        
        # Plot 5: Blue Win Rate
        ax = axes[2, 0]
        smoothed = self._smooth(self.metrics['blue_win_rate'])
        episodes_plot = np.arange(len(smoothed))
        ax.plot(episodes_plot, smoothed, color='green', linewidth=2)
        ax.set_xlabel('Episode', fontsize=11)
        ax.set_ylabel('Win Rate', fontsize=11)
        ax.set_title('Blue Win Rate (Attacks Blocked)', fontsize=12, fontweight='bold')
        ax.grid(True, alpha=0.3)
        
        # Plot 6: Red Epsilon
        ax = axes[2, 1]
        ax.plot(episodes, self.metrics['red_epsilon'], color='purple', linewidth=2)
        ax.set_xlabel('Episode', fontsize=11)
        ax.set_ylabel('Epsilon', fontsize=11)
        ax.set_title('Red Agent Exploration Rate', fontsize=12, fontweight='bold')
        ax.grid(True, alpha=0.3)
        
        # Hierarchical routing plots (if enabled)
        if self.use_hierarchy and len(self.metrics['tier1_only_rate']) > 0:
            ax = axes[3, 0]
            smoothed_t1 = self._smooth(self.metrics['tier1_only_rate'])
            smoothed_t2 = self._smooth(self.metrics['tier2_escalated_rate'])
            smoothed_t3 = self._smooth(self.metrics['tier3_analyzed_rate'])
            smoothed_ib = self._smooth(self.metrics['immediate_block_rate'])
            
            episodes_plot = np.arange(len(smoothed_t1))
            
            ax.plot(episodes_plot, smoothed_t1, label='Tier 1 Only', linewidth=2)
            ax.plot(episodes_plot[:len(smoothed_t2)], smoothed_t2, label='Tier 2 Escalated', linewidth=2)
            ax.plot(episodes_plot[:len(smoothed_t3)], smoothed_t3, label='Tier 3 Analyzed', linewidth=2)
            ax.plot(episodes_plot[:len(smoothed_ib)], smoothed_ib, label='Immediate Block', linewidth=2)
            ax.set_xlabel('Episode', fontsize=11)
            ax.set_ylabel('Routing Rate', fontsize=11)
            ax.set_title('Hierarchical Routing Distribution', fontsize=12, fontweight='bold')
            ax.legend()
            ax.grid(True, alpha=0.3)
            
            # Final routing pie chart
            ax = axes[3, 1]
            recent = min(100, len(self.metrics['tier1_only_rate']))
            final_routing = [
                np.mean(self.metrics['tier1_only_rate'][-recent:]),
                np.mean(self.metrics['tier2_escalated_rate'][-recent:]),
                np.mean(self.metrics['tier3_analyzed_rate'][-recent:]),
                np.mean(self.metrics['immediate_block_rate'][-recent:]),
            ]
            labels = ['Tier 1 Only', 'Tier 2\nEscalated', 'Tier 3\nAnalyzed', 'Immediate\nBlock']
            colors = ['#3498db', '#e74c3c', '#f39c12', '#2ecc71']
            ax.pie(final_routing, labels=labels, colors=colors, autopct='%1.1f%%', startangle=90)
            ax.set_title('Final Routing Distribution', fontsize=12, fontweight='bold')
        
        plt.tight_layout()
        
        filepath = f"{self.save_dir}/plots/training_results.png"
        plt.savefig(filepath, dpi=300, bbox_inches='tight')
        print(f"Training plots saved to {filepath}")
        
        plt.close()
    
    def _smooth(self, data, window=50):
        """Smooth data for plotting"""
        if len(data) == 0:
            return []
        if len(data) < window:
            return data
        
        smoothed = np.convolve(data, np.ones(window)/window, mode='same')
        return smoothed


# Main execution
if __name__ == "__main__":
    trainer = HierarchicalAdversarialTrainer(
        n_episodes=1000,
        max_steps_per_episode=500,
        save_freq=200,
        eval_freq=50,
        use_hierarchy=True,
        curriculum_learning=True
    )
    
    trainer.train()