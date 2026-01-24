"""
SQL Injection Dataset Loader
Generates attack and benign queries with 127-dimensional feature extraction
Implements feature engineering from Section 6.3 (Tier 1)
"""

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from collections import Counter
import re

class SQLInjectionDataset:
    """
    Dataset loader for SQL injection attacks and benign queries
    Features: 127 dimensions as specified in framework
    """
    
    def __init__(self, n_attacks=5000, n_benign=5000):
        self.n_attacks = n_attacks
        self.n_benign = n_benign
        self.attack_types = self._get_attack_taxonomy()
        
    def _get_attack_taxonomy(self):
        """
        20 SQL injection variants (subset of 47 from framework)
        Expandable to full taxonomy in future work
        """
        return {
            # Boolean-based blind (5 variants)
            'bool_1': "' OR '1'='1",
            'bool_2': "' OR 1=1--",
            'bool_3': "admin' OR '1'='1'--",
            'bool_4': "' OR 'x'='x",
            'bool_5': "1' OR '1'='1' AND '1'='1",
            
            # Union-based (4 variants)
            'union_1': "' UNION SELECT NULL--",
            'union_2': "' UNION SELECT username, password FROM users--",
            'union_3': "1' UNION ALL SELECT NULL,NULL,NULL--",
            'union_4': "' UNION SELECT @@version--",
            
            # Time-based blind (3 variants)
            'time_1': "'; WAITFOR DELAY '00:00:05'--",
            'time_2': "' OR SLEEP(5)--",
            'time_3': "1' AND SLEEP(5)--",
            
            # Error-based (3 variants)
            'error_1': "' AND 1=CONVERT(int, (SELECT @@version))--",
            'error_2': "' AND 1=CAST((SELECT TOP 1 name FROM sysobjects) AS INT)--",
            'error_3': "' AND extractvalue(1,concat(0x7e,version()))--",
            
            # Stacked queries (2 variants)
            'stack_1': "'; DROP TABLE users--",
            'stack_2': "'; INSERT INTO users VALUES('hacker','pass')--",
            
            # Comment-based (3 variants)
            'comment_1': "admin'--",
            'comment_2': "' OR 1=1#",
            'comment_3': "admin'/*",
        }
    
    def generate_dataset(self):
        """Generate complete dataset with labels"""
        print("Generating SQL injection dataset...")
        
        attacks = self._generate_attacks()
        benign = self._generate_benign()
        
        # Combine and shuffle
        all_data = attacks + benign
        np.random.shuffle(all_data)
        
        df = pd.DataFrame(all_data)
        print(f"Generated {len(df)} queries ({len(attacks)} attacks, {len(benign)} benign)")
        
        return df
    
    def _generate_attacks(self):
        """Generate attack queries with variations"""
        attacks = []
        attack_names = list(self.attack_types.keys())
        
        for _ in range(self.n_attacks):
            # Select random attack type
            attack_type = np.random.choice(attack_names)
            base_payload = self.attack_types[attack_type]
            
            # Add obfuscation
            payload = self._apply_obfuscation(base_payload)
            
            attacks.append({
                'query': payload,
                'label': 1,  # Malicious
                'attack_type': attack_type.split('_')[0],  # bool, union, time, etc.
                'variant': attack_type
            })
        
        return attacks
    
    def _generate_benign(self):
        """Generate legitimate SQL queries"""
        benign_templates = [
            "SELECT * FROM products WHERE id={id}",
            "SELECT name, price FROM items WHERE category='{cat}'",
            "INSERT INTO users (name, email) VALUES ('{name}', '{email}')",
            "UPDATE products SET price={price} WHERE id={id}",
            "DELETE FROM cart WHERE user_id={id}",
            "SELECT COUNT(*) FROM orders WHERE date > '{date}'",
            "SELECT * FROM customers WHERE name LIKE '{name}%'",
            "INSERT INTO logs (message, timestamp) VALUES ('{msg}', NOW())",
            "UPDATE users SET last_login=NOW() WHERE id={id}",
            "SELECT AVG(price) FROM products WHERE category='{cat}'",
        ]
        
        benign = []
        for _ in range(self.n_benign):
            template = np.random.choice(benign_templates)
            
            # Fill template with safe values
            query = template.format(
                id=np.random.randint(1, 1000),
                cat=np.random.choice(['electronics', 'books', 'clothing']),
                name=np.random.choice(['John', 'Alice', 'Bob', 'Sarah']),
                email='user@example.com',
                price=round(np.random.uniform(10, 1000), 2),
                date='2024-01-01',
                msg='User logged in'
            )
            
            benign.append({
                'query': query,
                'label': 0,  # Benign
                'attack_type': 'benign',
                'variant': 'legitimate'
            })
        
        return benign
    
    def _apply_obfuscation(self, payload):
        """Apply obfuscation techniques to make attacks more realistic"""
        techniques = [
            lambda p: p,  # No obfuscation (40% of time)
            lambda p: p.replace(' ', '/**/'),  # Comment injection
            lambda p: p.upper() if np.random.random() > 0.5 else p.lower(),  # Case variation
            lambda p: p.replace("'", "%27").replace(' ', '%20'),  # URL encoding
            lambda p: p.replace('=', '%3D').replace("'", "%27"),  # Partial encoding
        ]
        
        # 60% chance of obfuscation
        if np.random.random() < 0.6:
            technique = np.random.choice(techniques[1:])  # Exclude no-obfuscation
        else:
            technique = techniques[0]
        
        return technique(payload)
    
    def extract_features(self, query):
        """
        Extract 127-dimensional feature vector
        Implements Section 6.3 Tier 1 observation space
        """
        features = []
        query_upper = query.upper()
        
        # === TOKEN-LEVEL ANALYSIS (42 dimensions) ===
        sql_keywords = [
            'SELECT', 'UNION', 'WHERE', 'OR', 'AND', 'DROP', 'INSERT',
            'UPDATE', 'DELETE', 'FROM', 'TABLE', 'ALTER', 'CREATE',
            'EXEC', 'EXECUTE', 'DECLARE', 'CAST', 'CONVERT', 'CHAR',
            'NCHAR', 'VARCHAR', 'NVARCHAR', 'CONCAT', 'SUBSTRING',
            'WAITFOR', 'DELAY', 'SLEEP', 'BENCHMARK', 'EXTRACTVALUE',
            'UPDATEXML', 'LOAD_FILE', 'INTO', 'OUTFILE', 'DUMPFILE',
            'INFORMATION_SCHEMA', 'SYSOBJECTS', 'SYSCOLUMNS', 'VERSION',
            'USER', 'DATABASE', 'SCHEMA'
        ]
        
        for keyword in sql_keywords[:42]:  # Take first 42
            features.append(query_upper.count(keyword))
        
        # === SPECIAL CHARACTERS (30 dimensions) ===
        special_chars = [
            "'", '"', '--', '#', ';', '/*', '*/', '=', '<', '>',
            '(', ')', ',', '%', '_', '&', '|', '^', '!', '@',
            '$', '+', '-', '*', '/', '\\', '[', ']', '{', '}'
        ]
        
        for char in special_chars:
            features.append(query.count(char))
        
        # === SYNTACTIC STRUCTURE (35 dimensions) ===
        features.extend([
            query.count('('),  # Parenthesis nesting
            query.count(')'),
            query.count('SELECT'),  # Clause count
            query.count('FROM'),
            query.count('WHERE'),
            query.count('UNION'),
            query.count('JOIN'),
            len(query),  # Query length
            len(query.split()),  # Word count
            query.count(' OR '),  # Logic operators
            query.count(' AND '),
            query.count('='),  # Comparison operators
            query.count('<'),
            query.count('>'),
            query.count('LIKE'),
            # Padding to 35
            *[0] * 20
        ])
        
        # === CHARACTER DISTRIBUTION (25 dimensions) ===
        entropy = self._calculate_entropy(query)
        features.append(entropy)
        
        char_freqs = self._get_char_frequencies(query)
        features.extend(char_freqs[:24])  # Top 24 character frequencies
        
        # === ENCODING IDENTIFICATION (15 dimensions) ===
        features.extend([
            int('%' in query),  # URL encoding present
            query.count('%'),  # URL encoding count
            int('0x' in query),  # Hex encoding
            int('\\x' in query or '\\u' in query),  # Unicode
            int('CHAR(' in query_upper),  # Character encoding functions
            int('CHR(' in query_upper),
            int('ASCII(' in query_upper),
            int('HEX(' in query_upper),
            int('UNHEX(' in query_upper),
            int('BASE64' in query_upper),
            *[0] * 5  # Padding
        ])
        
        # Ensure exactly 127 dimensions
        features = features[:127]
        while len(features) < 127:
            features.append(0)
        
        return np.array(features, dtype=np.float32)
    
    def _calculate_entropy(self, text):
        """Calculate Shannon entropy"""
        if not text:
            return 0.0
        
        counter = Counter(text)
        length = len(text)
        entropy = -sum((count/length) * np.log2(count/length) 
                      for count in counter.values())
        return entropy
    
    def _get_char_frequencies(self, text):
        """Get top character frequencies"""
        if not text:
            return [0.0] * 50
        
        counter = Counter(text)
        total = len(text)
        freqs = [count/total for char, count in counter.most_common(50)]
        
        while len(freqs) < 50:
            freqs.append(0.0)
        
        return freqs[:50]
    
    def get_train_test_split(self, df, test_size=0.2):
        """Split with stratification"""
        train_df, test_df = train_test_split(
            df, 
            test_size=test_size, 
            stratify=df['label'],
            random_state=42
        )
        
        print(f"Train: {len(train_df)} | Test: {len(test_df)}")
        return train_df, test_df


# Quick test
if __name__ == "__main__":
    dataset = SQLInjectionDataset(n_attacks=100, n_benign=100)
    df = dataset.generate_dataset()
    
    # Test feature extraction
    sample = df.iloc[0]
    features = dataset.extract_features(sample['query'])
    print(f"\nSample query: {sample['query']}")
    print(f"Label: {sample['label']}")
    print(f"Features shape: {features.shape}")
    print(f"Feature sample: {features[:10]}")
