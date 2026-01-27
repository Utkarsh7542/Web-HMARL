"""
Baseline Comparisons
Implements ModSecurity CRS-style rule-based detection
For comparison with Web-HMARL
"""

import re
import numpy as np
import sys
import os

class ModSecurityBaseline:
    """
    ModSecurity CRS-inspired baseline
    Rule-based SQL injection detection
    """
    
    def __init__(self):
        # Common SQL injection patterns (from OWASP ModSecurity CRS)
        self.patterns = [
            r"(\bUNION\b.*\bSELECT\b)",
            r"(\bOR\b\s+\d+\s*=\s*\d+)",
            r"('|\")(\s*OR\s*'1'\s*=\s*'1)",
            r"(;|\|{2})\s*(DROP|DELETE|INSERT|UPDATE)\b",
            r"(--|\#|\/\*)",
            r"(\bWAITFOR\b|\bSLEEP\b|\bBENCHMARK\b)",
            r"(\bEXEC\b|\bEXECUTE\b)\s*\(",
            r"(\bINFORMATION_SCHEMA\b)",
            r"(\bSYSOBJECTS\b|\bSYSCOLUMNS\b)",
            r"(CAST\s*\(|CONVERT\s*\()",
            r"(\bLOAD_FILE\b|\bINTO\s+OUTFILE\b)",
        ]
        
        self.compiled_patterns = [re.compile(p, re.IGNORECASE) for p in self.patterns]
        
        # Statistics
        self.total_queries = 0
        self.blocked = 0
        self.correct_blocks = 0
        self.false_positives = 0
        self.missed_attacks = 0
    
    def detect(self, query):
        """
        Detect SQL injection using rule-based patterns
        
        Returns:
            is_malicious: bool
            matched_patterns: list of pattern indices
        """
        self.total_queries += 1
        matched = []
        
        for i, pattern in enumerate(self.compiled_patterns):
            if pattern.search(query):
                matched.append(i)
        
        # If any pattern matches → malicious
        is_malicious = len(matched) > 0
        
        if is_malicious:
            self.blocked += 1
        
        return is_malicious, matched
    
    def evaluate(self, query, true_label):
        """
        Evaluate detection and update statistics
        
        Args:
            query: SQL query string
            true_label: 1 if attack, 0 if benign
        """
        detected, _ = self.detect(query)
        
        if true_label == 1:  # Actual attack
            if detected:
                self.correct_blocks += 1
            else:
                self.missed_attacks += 1
        else:  # Benign query
            if detected:
                self.false_positives += 1
    
    def get_metrics(self):
        """Calculate performance metrics"""
        total_attacks = self.correct_blocks + self.missed_attacks
        total_benign = self.total_queries - total_attacks
        
        if total_attacks == 0:
            detection_rate = 0.0
        else:
            detection_rate = self.correct_blocks / total_attacks
        
        if total_benign == 0:
            fp_rate = 0.0
        else:
            fp_rate = self.false_positives / total_benign
        
        return {
            'detection_rate': detection_rate * 100,
            'false_positive_rate': fp_rate * 100,
            'total_blocks': self.blocked,
            'correct_blocks': self.correct_blocks,
            'false_positives': self.false_positives,
            'missed_attacks': self.missed_attacks
        }
    
    def reset_stats(self):
        """Reset statistics"""
        self.total_queries = 0
        self.blocked = 0
        self.correct_blocks = 0
        self.false_positives = 0
        self.missed_attacks = 0


class SimpleMLBaseline:
    """
    Simple ML baseline (logistic regression on features)
    Represents ModSec-AdvLearn simplified approach
    """
    
    def __init__(self):
        from sklearn.linear_model import LogisticRegression
        self.model = LogisticRegression(max_iter=1000)
        self.is_trained = False
    
    def train(self, X_train, y_train):
        """Train on labeled data"""
        self.model.fit(X_train, y_train)
        self.is_trained = True
    
    def predict(self, X):
        """Predict if queries are malicious"""
        if not self.is_trained:
            raise ValueError("Model not trained!")
        return self.model.predict(X)
    
    def evaluate(self, X_test, y_test):
        """Evaluate performance"""
        from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
        
        predictions = self.predict(X_test)
        
        return {
            'accuracy': accuracy_score(y_test, predictions) * 100,
            'precision': precision_score(y_test, predictions, zero_division=0) * 100,
            'recall': recall_score(y_test, predictions, zero_division=0) * 100,
            'f1_score': f1_score(y_test, predictions, zero_division=0) * 100
        }

# Test
if __name__ == "__main__":
    # Add project root to Python path for imports
    project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    if project_root not in sys.path:
        sys.path.insert(0, project_root)
    
    print("Testing ModSecurity Baseline...")
    
    baseline = ModSecurityBaseline()
    
    # Test attacks
    attacks = [
        "SELECT * FROM users WHERE id=1 OR 1=1",
        "' UNION SELECT password FROM admin--",
        "'; DROP TABLE users--",
    ]
    
    benign = [
        "SELECT * FROM products WHERE id=5",
        "INSERT INTO logs VALUES ('user login')",
    ]
    
    print("\nTesting attacks:")
    for attack in attacks:
        detected, patterns = baseline.detect(attack)
        baseline.evaluate(attack, true_label=1)
        print(f"  '{attack[:40]}...' → Detected: {detected}")
    
    print("\nTesting benign:")
    for query in benign:
        detected, patterns = baseline.detect(query)
        baseline.evaluate(query, true_label=0)
        print(f"  '{query[:40]}...' → Detected: {detected}")
    
    metrics = baseline.get_metrics()
    print(f"\nMetrics: {metrics}")
    
    print("\n--- Testing Simple ML Baseline ---")
    from data.sqli_dataset import SQLInjectionDataset
    
    dataset_loader = SQLInjectionDataset(n_attacks=100, n_benign=100)
    df = dataset_loader.generate_dataset()
    
    # Extract features
    X = np.array([dataset_loader.extract_features(q) for q in df['query']])
    y = df['label'].values
    
    # Split
    from sklearn.model_selection import train_test_split
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3)
    
    # Train and evaluate
    ml_baseline = SimpleMLBaseline()
    ml_baseline.train(X_train, y_train)
    ml_metrics = ml_baseline.evaluate(X_test, y_test)
    
    print(f"ML Baseline metrics: {ml_metrics}")
    
    print("\n✅ Baseline test complete!")