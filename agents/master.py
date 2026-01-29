"""
Master Coordination Layer - IMPROVED VERSION
✅ Adaptive routing based on system state
✅ Lower thresholds to actually use hierarchy
✅ Dynamic tier selection based on threat level
✅ Better load balancing across tiers
"""

import numpy as np

class MasterCoordinator:
    """
    Master Policy: Expert-Driven Coordination with Adaptive Routing
    
    IMPROVEMENTS:
    - Lower confidence thresholds (actually reachable)
    - Adaptive routing percentages based on attack rate
    - Dynamic tier selection based on system threat level
    - Better metrics tracking
    """
    
    def __init__(self):
        # Thresholds (LOWERED to be reachable with rebalanced blue)
        self.thresholds = {
            'pass_threshold': 0.25,      # LOWERED from 0.3
            'escalate_threshold': 0.55,  # LOWERED from 0.6
            'suggest_threshold': 0.85,   # LOWERED from 0.9
            'block_threshold': 0.85      # LOWERED from 0.9
        }
        
        # Adaptive routing parameters
        self.base_tier2_rate = 0.30     # Base 30% to Tier 2 (up from 20%)
        self.min_tier2_rate = 0.20      # Minimum 20%
        self.max_tier2_rate = 0.50      # Maximum 50%
        
        # System state tracking
        self.system_state = {
            'recent_attack_rate': 0.0,
            'recent_fp_rate': 0.0,
            'system_load': 0.0,
            'threat_level': 'low',  # low, medium, high, critical
            'total_queries': 0,
            'attacks_detected': 0,
        }
        
        # Statistics
        self.routing_stats = {
            'tier1_only': 0,
            'tier2_escalated': 0,
            'tier3_analyzed': 0,
            'immediate_blocks': 0
        }
        
        # Confidence history for adaptive thresholds
        self.confidence_history = []
        
    def coordinate(self, tier1_confidence, tier1_action, query_features):
        """
        Master coordination decision - ADAPTIVE VERSION
        
        Returns:
            route: 'tier1_only', 'tier2', 'tier3', or 'immediate_block'
            final_action: int (0-3) or None if needs escalation
            confidence: float
        """
        
        # Track confidence distribution
        self.confidence_history.append(tier1_confidence)
        if len(self.confidence_history) > 1000:
            self.confidence_history.pop(0)
        
        # Calculate adaptive Tier 2 rate based on threat level
        tier2_rate = self._get_adaptive_tier2_rate()
        
        # ADAPTIVE RULE 1: Force percentage to Tier 2 based on threat
        if np.random.random() < tier2_rate:
            self.routing_stats['tier2_escalated'] += 1
            return 'tier2', None, tier1_confidence
        
        # RULE 2: High confidence benign → Pass directly
        # LOWERED threshold from 0.92 to 0.80
        if tier1_confidence > 0.80 and tier1_action == 0:
            self.routing_stats['tier1_only'] += 1
            return 'tier1_only', tier1_action, tier1_confidence
        
        # RULE 3: VERY high threat → Immediate block
        # LOWERED threshold from 0.97 to 0.90
        if tier1_confidence > 0.90 and tier1_action == 3:
            self.routing_stats['immediate_blocks'] += 1
            return 'immediate_block', 3, tier1_confidence
        
        # RULE 4: Medium confidence → Escalate to Tier 2
        # EXPANDED range: 0.15-0.80 (was 0.2-0.92)
        if 0.15 <= tier1_confidence <= 0.80:
            self.routing_stats['tier2_escalated'] += 1
            return 'tier2', None, tier1_confidence
        
        # RULE 5: Low confidence + suspicious patterns → Tier 3 analysis
        # INCREASED rate from 5% to 10%
        if tier1_confidence < 0.15:
            if self._is_suspicious(query_features) or np.random.random() < 0.10:
                self.routing_stats['tier3_analyzed'] += 1
                return 'tier3', tier1_action, tier1_confidence
        
        # RULE 6: Critical threat level → More Tier 2 escalation
        if self.system_state['threat_level'] in ['high', 'critical']:
            if np.random.random() < 0.4:  # 40% during high threat
                self.routing_stats['tier2_escalated'] += 1
                return 'tier2', None, tier1_confidence
        
        # Default: Use Tier 1 decision
        self.routing_stats['tier1_only'] += 1
        return 'tier1_only', tier1_action, tier1_confidence
    
    def _get_adaptive_tier2_rate(self):
        """
        Calculate adaptive Tier 2 routing rate based on system state
        
        Returns:
            float: Percentage of queries to route to Tier 2
        """
        base_rate = self.base_tier2_rate
        
        # Increase Tier 2 usage during high attack rate
        if self.system_state['recent_attack_rate'] > 0.3:
            base_rate += 0.10  # +10% during attacks
        elif self.system_state['recent_attack_rate'] > 0.5:
            base_rate += 0.15  # +15% during heavy attacks
        
        # Increase Tier 2 usage if false positive rate is high
        if self.system_state['recent_fp_rate'] > 0.2:
            base_rate += 0.05  # +5% to reduce FPs with deeper analysis
        
        # Decrease Tier 2 usage under high load (hypothetical)
        if self.system_state['system_load'] > 0.8:
            base_rate -= 0.05  # -5% under load
        
        # Clamp to valid range
        return np.clip(base_rate, self.min_tier2_rate, self.max_tier2_rate)
    
    def _is_suspicious(self, query_features):
        """
        Heuristic: Check for suspicious patterns
        Enhanced version with more sophisticated checks
        """
        # Check SQL keyword concentration (first 42 features)
        sql_keyword_score = np.sum(query_features[:42])
        
        # Check special character usage (features 42-84)
        special_char_score = np.sum(query_features[42:84])
        
        # Check length and complexity (features 84+)
        complexity_score = np.sum(query_features[84:])
        
        # Combine scores with weights
        suspicion_score = (
            sql_keyword_score * 0.5 +
            special_char_score * 0.3 +
            complexity_score * 0.2
        )
        
        # Dynamic threshold based on recent attack rate
        threshold = 5.0
        if self.system_state['recent_attack_rate'] > 0.3:
            threshold = 4.0  # More sensitive during attack waves
        
        return suspicion_score > threshold
    
    def update_system_state(self, attack_detected, false_positive):
        """
        Update system state based on recent activity
        
        Args:
            attack_detected: bool, true positive detection
            false_positive: bool, false positive
        """
        # Simple exponential moving average
        alpha = 0.1
        
        self.system_state['total_queries'] += 1
        
        if attack_detected:
            self.system_state['attacks_detected'] += 1
            self.system_state['recent_attack_rate'] = (
                alpha * 1.0 + (1 - alpha) * self.system_state['recent_attack_rate']
            )
        else:
            self.system_state['recent_attack_rate'] = (
                alpha * 0.0 + (1 - alpha) * self.system_state['recent_attack_rate']
            )
        
        if false_positive:
            self.system_state['recent_fp_rate'] = (
                alpha * 1.0 + (1 - alpha) * self.system_state['recent_fp_rate']
            )
        else:
            self.system_state['recent_fp_rate'] = (
                alpha * 0.0 + (1 - alpha) * self.system_state['recent_fp_rate']
            )
        
        # Update threat level based on attack rate
        attack_rate = self.system_state['recent_attack_rate']
        if attack_rate > 0.6:
            self.system_state['threat_level'] = 'critical'
        elif attack_rate > 0.4:
            self.system_state['threat_level'] = 'high'
        elif attack_rate > 0.2:
            self.system_state['threat_level'] = 'medium'
        else:
            self.system_state['threat_level'] = 'low'
        
        # Simulate system load (in real system, would be actual metrics)
        # For now, based on query volume and threat level
        if self.system_state['total_queries'] % 100 == 0:
            base_load = min(0.9, self.system_state['total_queries'] / 10000)
            threat_load = {'low': 0.0, 'medium': 0.1, 'high': 0.2, 'critical': 0.3}
            self.system_state['system_load'] = min(1.0, base_load + threat_load[self.system_state['threat_level']])
    
    def get_context_for_tier2(self):
        """
        Provide context to Tier 2
        Returns: (4,) array [user_reputation, endpoint_criticality, system_load, recent_fp_rate]
        
        Enhanced with actual system state
        """
        return np.array([
            0.5,  # user_reputation (placeholder - would be real in production)
            0.7,  # endpoint_criticality (placeholder - would be real in production)
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
    
    def get_system_state_summary(self):
        """Get human-readable system state"""
        return {
            'threat_level': self.system_state['threat_level'],
            'attack_rate': f"{self.system_state['recent_attack_rate']:.2%}",
            'fp_rate': f"{self.system_state['recent_fp_rate']:.2%}",
            'system_load': f"{self.system_state['system_load']:.2%}",
            'total_queries': self.system_state['total_queries'],
            'tier2_routing': f"{self._get_adaptive_tier2_rate():.2%}",
        }
    
    def reset_stats(self):
        """Reset routing statistics (for new episode/evaluation)"""
        self.routing_stats = {
            'tier1_only': 0,
            'tier2_escalated': 0,
            'tier3_analyzed': 0,
            'immediate_blocks': 0
        }


# Test
if __name__ == "__main__":
    print("Testing Master Coordinator (Improved)...")
    
    master = MasterCoordinator()
    
    print(f"Initial Tier 2 rate: {master._get_adaptive_tier2_rate():.2%}")
    print(f"Thresholds: {master.thresholds}")
    
    # Simulate different scenarios
    test_query = np.random.randn(127)
    
    print("\n=== Test 1: High confidence benign ===")
    route, action, conf = master.coordinate(
        tier1_confidence=0.85,
        tier1_action=0,
        query_features=test_query
    )
    print(f"Route: {route}, Action: {action}")
    
    print("\n=== Test 2: High threat ===")
    route, action, conf = master.coordinate(
        tier1_confidence=0.92,
        tier1_action=3,
        query_features=test_query
    )
    print(f"Route: {route}, Action: {action}")
    
    print("\n=== Test 3: Medium confidence (should escalate) ===")
    route, action, conf = master.coordinate(
        tier1_confidence=0.5,
        tier1_action=1,
        query_features=test_query
    )
    print(f"Route: {route}, Action: {action}")
    
    print("\n=== Test 4: Low confidence suspicious ===")
    suspicious_query = np.random.randn(127)
    suspicious_query[:42] = 10  # High SQL keyword score
    route, action, conf = master.coordinate(
        tier1_confidence=0.1,
        tier1_action=1,
        query_features=suspicious_query
    )
    print(f"Route: {route}, Action: {action}")
    
    # Simulate attack wave
    print("\n=== Simulating attack wave ===")
    for i in range(50):
        is_attack = i % 2 == 0  # 50% attack rate
        master.update_system_state(attack_detected=is_attack, false_positive=False)
    
    print(f"System state: {master.get_system_state_summary()}")
    print(f"Adaptive Tier 2 rate: {master._get_adaptive_tier2_rate():.2%}")
    
    # Get routing distribution
    for _ in range(100):
        conf = np.random.uniform(0.1, 0.95)
        action = np.random.randint(0, 4)
        master.coordinate(conf, action, test_query)
    
    stats = master.get_routing_stats()
    print(f"\n=== Routing Distribution (100 queries) ===")
    for route, pct in stats.items():
        print(f"  {route}: {pct:.1f}%")
    
    print("\n✅ Master coordinator test complete!")