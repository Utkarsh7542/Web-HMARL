"""
Main entry point for Web-HMARL training
Run quick test or full training
"""

import argparse
from training.adversarial_trainer import HierarchicalAdversarialTrainer

def main():
    parser = argparse.ArgumentParser(description='Web-HMARL Training')
    parser.add_argument('--episodes', type=int, default=1000, 
                       help='Number of training episodes')
    parser.add_argument('--steps', type=int, default=500,
                       help='Max steps per episode')
    parser.add_argument('--quick-test', action='store_true',
                       help='Run quick test (100 episodes)')
    
    args = parser.parse_args()
    
    if args.quick_test:
        print("Running quick test...")
        episodes = 100
        steps = 100
    else:
        episodes = args.episodes
        steps = args.steps
    
    # Create and run trainer
    trainer = HierarchicalAdversarialTrainer(
        n_episodes=episodes,
        max_steps_per_episode=steps,
        save_freq=max(50, episodes // 10),
        eval_freq=max(20, episodes // 20)
    )
    
    trainer.train()
    
    print("\n✅ Training complete! Check results/ folder for:")
    print("  - results/models/ (saved checkpoints)")
    print("  - results/plots/ (training graphs)")
    print("  - results/logs/ (metrics JSON)")

if __name__ == "__main__":
    main()
