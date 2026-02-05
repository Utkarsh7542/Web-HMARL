# Web-HMARL: SQL Injection Defense using Hierarchical Multi-Agent Reinforcement Learning

This repository contains the implementation of our research on using hierarchical multi-agent reinforcement learning to defend against SQL injection attacks.

## What is this?

Web-HMARL is a system where two AI agents play a continuous game:

- **Red Agent (Attacker)**: Learns to generate SQL injection attacks
- **Blue Agent (Defender)**: Learns to detect and block these attacks through a 3-tier defense system

Unlike traditional rule-based firewalls, both agents keep learning and adapting, creating a more realistic security scenario.

## Key Results

After training for 500 episodes:

- **Detection accuracy**: 87.8%
- **False positive rate**: 31.6%
- **60% of queries** handled by fast Tier 1 detection (12ms latency)
- **Hierarchical routing** reduces average latency by 50%

## How it works

### Red Agent (Attacker)

- Uses Deep Q-Network (DQN) to learn attack strategies
- Generates 30 types of SQL injection variants
- Employs mutation-based techniques after episode 300
- Action space: 30 SQL injection variants with obfuscation

### Blue Agent (Defender)

**Tier 1**: Fast Bi-LSTM detector (64 hidden units, 12ms latency)
- 4 actions: Pass, Escalate, Suggest Block, Immediate Block
- Handles 60% of queries without escalation

**Tier 2**: Deep analysis using feedforward network (35.6% of queries)
- Contextual analysis for uncertain cases

## Results Visualization

After training, check `results/plots/` for:

- `training_curves.png`: Episode rewards and convergence
- `routing_distribution.png`: Hierarchical routing efficiency
- `metrics.json`: Detailed performance metrics

## Limitations

- Currently focused on SQL injection only
- Trained on synthetic dataset (2,000 queries)
- False positive rate of 31.6% needs tuning for production
- Evaluation on controlled scenarios, not live traffic

## Future Work

- Extend to other web vulnerabilities (XSS, CSRF)
- Implement Tier 3 forensic analysis
- Test on public datasets like CICIDS2017
- Reduce false positive rate through threshold tuning



## Acknowledgments

- Built using PyTorch and Gymnasium frameworks
- Inspired by hierarchical MARL research in network security
- Thanks to our supervisor Dr. Aarti Sehwag for guidance

---

**Note**: This is research code. For production use, additional hardening and testing would be required.
