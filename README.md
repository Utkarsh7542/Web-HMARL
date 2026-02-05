Web-HMARL: SQL Injection Defense using Hierarchical Multi-Agent Reinforcement Learning
This repository contains the implementation of our research on using hierarchical multi-agent reinforcement learning to defend against SQL injection attacks.
What is this?
Web-HMARL is a system where two AI agents play a continuous game:

Red Agent (Attacker): Learns to generate SQL injection attacks
Blue Agent (Defender): Learns to detect and block these attacks through a 3-tier defense system

Unlike traditional rule-based firewalls, both agents keep learning and adapting, creating a more realistic security scenario.
Key Results
After training for 500 episodes:

Detection accuracy: 87.8%
False positive rate: 31.6%
60% of queries handled by fast Tier 1 detection (12ms latency)
Hierarchical routing reduces average latency by 50%
How it works
Red Agent (Attacker)

Uses Deep Q-Network (DQN) to learn attack strategies
Generates 30 types of SQL injection variants
Employs mutation-based techniques after episode 300
Action space: 30 SQL injection variants with obfuscation

Blue Agent (Defender)

Tier 1: Fast Bi-LSTM detector (64 hidden units, 12ms latency)

4 actions: Pass, Escalate, Suggest Block, Immediate Block
Handles 60% of queries without escalation


Tier 2: Deep analysis using feedforward network (35.6% of queries)

Contextual analysis for uncertain cases


Master Policy: Routes queries based on confidence thresholds

Training Process

Phase 0 (Episodes 0-150): Blue gets 30% less training to avoid early dominance
Phase 1 (Episodes 150-300): Standard training for both agents
Phase 2 (Episodes 300+): Red agent mutation capabilities enabled
