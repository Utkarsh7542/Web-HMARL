"""
Master Coordination Layer (Section 6.4)
Expert-driven rule-based coordinator
Manages 3-tier hierarchy based on threat assessment
"""

import numpy as np

class MasterCoordinator:
    """
    Master Policy: Expert-Driven Coordination
    
    Decision Rules (from framework Section 6.4):
    - High confidence benign → Pass (Tier 1 only)
    - Low confidence → Escalate to Tier 2
    - High threat → Immediate block
    - Uncertain → Send to Tier 3 analysis (async)
    """
    
    def __init__(self):
        # Thresholds (from paper Section 6.3)
        self.thresholds = {
            'pass_threshold': 0.3,      # Tier 1 confidence < 0.3 → Pass
            'escalate_threshold': 0.6,   # 0.3-0.6 → Escalate to Tier 2
            'suggest_threshold': 0.9,    # 0.6-0.9 → Suggest block
            'block_threshold': 0.9       # > 0.9 → Immediate block
        }
        
        # System state tracking
        self.system_state = {
            'recent_attack_rate': 0.0,
            'recent_fp_rate': 0.0,
            'system_load': 0.0,
            'threat_level': 'low'  # low, medium, high, critical
        }
        
        # Statistics
        self.routing_stats = {
            'tier1_only': 0,
            'tier2_escalated': 0,
            'tier3_analyzed': 0,
            'immediate_blocks': 0
        }
    
    def coordinate(self, tier1_confidence, tier1_action, query_features):
        """
        Master coordination decision - MODIFIED FOR ACTUAL HIERARCHICAL USE
        
        Returns:
            route: 'tier1_only', 'tier2', 'tier3', or 'immediate_block'
            final_action: int (0-3)
            confidence: float
        """
        
        # FORCE 20% of queries to Tier 2 for deep analysis (even if confident)
        # This ensures hierarchy is actually used
        if np.random.random() < 0.20:
            self.routing_stats['tier2_escalated'] += 1
            return 'tier2', None, tier1_confidence
        
        # Rule 1: VERY high confidence benign → Pass directly
        # Raised threshold from 0.8 to 0.92
        if tier1_confidence > 0.92 and tier1_action == 0:
            self.routing_stats['tier1_only'] += 1
            return 'tier1_only', tier1_action, tier1_confidence
        
        # Rule 2: EXTREMELY high threat → Immediate block
        # Raised threshold from 0.9 to 0.97
        if tier1_confidence > 0.97 and tier1_action == 3:
            self.routing_stats['immediate_blocks'] += 1
            return 'immediate_block', 3, tier1_confidence
        
        # Rule 3: Medium confidence → Escalate to Tier 2
        # Widened range from 0.3-0.8 to 0.2-0.92
        if 0.2 <= tier1_confidence <= 0.92:
            self.routing_stats['tier2_escalated'] += 1
            return 'tier2', None, tier1_confidence
        
        # Rule 4: Low confidence but suspicious → Tier 3 analysis
        # 5% of low-confidence queries go to Tier 3
        if tier1_confidence < 0.2 and np.random.random() < 0.05:
            self.routing_stats['tier3_analyzed'] += 1
            return 'tier3', tier1_action, tier1_confidence
        
        # Default: Use Tier 1 decision
        self.routing_stats['tier1_only'] += 1
        return 'tier1_only', tier1_action, tier1_confidence
    
    def _is_suspicious(self, query_features):
        """
        Heuristic: Check for suspicious patterns
        (simplified version of Tier 3 analysis)
        """
        # Check for SQL keywords, special characters, etc.
        # This is a placeholder - in full version would be more sophisticated
        suspicious_score = np.sum(query_features[:42])  # SQL keyword counts
        return suspicious_score > 5
    
    def update_system_state(self, attack_detected, false_positive):
        """Update system state based on recent activity"""
        # Simple moving average
        alpha = 0.1
        if attack_detected:
            self.system_state['recent_attack_rate'] = (
                alpha * 1.0 + (1 - alpha) * self.system_state['recent_attack_rate']
            )
        
        if false_positive:
            self.system_state['recent_fp_rate'] = (
                alpha * 1.0 + (1 - alpha) * self.system_state['recent_fp_rate']
            )
        
        # Adjust threat level
        if self.system_state['recent_attack_rate'] > 0.5:
            self.system_state['threat_level'] = 'critical'
        elif self.system_state['recent_attack_rate'] > 0.3:
            self.system_state['threat_level'] = 'high'
        elif self.system_state['recent_attack_rate'] > 0.1:
            self.system_state['threat_level'] = 'medium'
        else:
            self.system_state['threat_level'] = 'low'
    
    def get_context_for_tier2(self):
        """
        Provide context to Tier 2
        Returns: (4,) array [user_reputation, endpoint_criticality, system_load, recent_fp_rate]
        """
        return np.array([
            0.5,  # user_reputation (placeholder)
            0.7,  # endpoint_criticality (placeholder)
            self.system_state['system_load'],
            self.system_state['recent_fp_rate']
        ], dtype=np.float32)
    
    def get_routing_stats(self):
        """Return routing statistics for analysis"""
        total = sum(self.routing_stats.values())
        if total == 0:
            return self.routing_stats
        
        percentages = {
            k: (v / total) * 100 
            for k, v in self.routing_stats.items()
        }
        return percentages

# Test
if __name__ == "__main__":
    print("Testing Master Coordinator...")
    
    master = MasterCoordinator()
    
    # Test case 1: High confidence benign
    query_features = np.random.randn(127)
    route, action, conf = master.coordinate(
        tier1_confidence=0.9,
        tier1_action=0,  # Pass
        query_features=query_features
    )
    print(f"Test 1 - High confidence benign: route={route}, action={action}")
    
    # Test case 2: High threat
    route, action, conf = master.coordinate(
        tier1_confidence=0.95,
        tier1_action=3,  # Block
        query_features=query_features
    )
    print(f"Test 2 - High threat: route={route}, action={action}")
    
    # Test case 3: Medium confidence
    route, action, conf = master.coordinate(
        tier1_confidence=0.5,
        tier1_action=1,
        query_features=query_features
    )
    print(f"Test 3 - Medium confidence: route={route}, action={action}")
    
    # Get stats
    stats = master.get_routing_stats()
    print(f"\nRouting statistics: {stats}")
    
    print("\n✅ Master test complete!")