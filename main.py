"""
Main entry point for Web-HMARL training
Run quick test or full training
"""

import argparse
import torch
from training.adversarial_trainer import AdversarialTrainer

def main():
    parser = argparse.ArgumentParser(description='Web-HMARL Training')
    parser.add_argument('--episodes', type=int, default=1000, 
                       help='Number of training episodes')
    parser.add_argument('--steps', type=int, default=500,
                       help='Max steps per episode')
    parser.add_argument('--quick-test', action='store_true',
                       help='Run quick test (100 episodes)')
    parser.add_argument('--cpu', action='store_true',
                       help='Force CPU usage (disable GPU)')
    parser.add_argument('--gpu-id', type=int, default=0,
                       help='GPU ID to use (default: 0)')
    
    args = parser.parse_args()
    
    if args.quick_test:
        print("Running quick test...")
        episodes = 100
        steps = 100
    else:
        episodes = args.episodes
        steps = args.steps
    
    # Device selection
    use_gpu = not args.cpu
    device = None
    if use_gpu and torch.cuda.is_available():
        if args.gpu_id < torch.cuda.device_count():
            device = torch.device(f'cuda:{args.gpu_id}')
        else:
            device = torch.device('cuda:0')
            print(f"⚠️  GPU {args.gpu_id} not available, using GPU 0")
    else:
        device = torch.device('cpu')
    
    # Create and run trainer
    trainer = AdversarialTrainer(
        n_episodes=episodes,
        max_steps_per_episode=steps,
        save_freq=max(50, episodes // 10),
        eval_freq=max(20, episodes // 20),
        device=device,
        use_gpu=use_gpu
    )
    
    trainer.train()
    
    print("\n✅ Training complete! Check results/ folder for:")
    print("  - results/models/ (saved checkpoints)")
    print("  - results/plots/ (training graphs)")
    print("  - results/logs/ (metrics JSON)")

if __name__ == "__main__":
    main()
