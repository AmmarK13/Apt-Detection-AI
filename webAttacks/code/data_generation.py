import pandas as pd
import random
import string
import os

def random_string(length):
    """Generate a random string of specified length"""
    return ''.join(random.choice(string.ascii_letters + string.digits) for _ in range(length))

def create_attack_request(attack_type):
    """Create different types of attack requests"""
    if attack_type == 'sql_injection':
        payloads = [
            "' OR 1=1 --",
            "'; DROP TABLE users; --",
            "' UNION SELECT username, password FROM users --",
            "1' OR '1'='1",
            "admin' --",
            "1; SELECT * FROM information_schema.tables"
        ]
        url = f"/search?q={random.choice(payloads)}"
        content = f"username={random.choice(payloads)}"
        
    elif attack_type == 'xss':
        payloads = [
            "<script>alert('XSS')</script>",
            "<img src='x' onerror='alert(1)'>",
            "javascript:alert(document.cookie)",
            "<svg onload='alert(1)'>",
            "'\"><script>alert(1)</script>"
        ]
        url = f"/page?id={random.choice(payloads)}"
        content = f"comment={random.choice(payloads)}"
        
    elif attack_type == 'path_traversal':
        payloads = [
            "../../../etc/passwd",
            "..\\..\\..\\windows\\system32\\config\\sam",
            "file:///etc/passwd",
            "/var/www/html/config.php",
            "../../.env"
        ]
        url = f"/download?file={random.choice(payloads)}"
        content = ""
        
    elif attack_type == 'command_injection':
        payloads = [
            "| cat /etc/passwd",
            "; rm -rf /",
            "& net user",
            "|| whoami",
            "`ping -c 4 attacker.com`"
        ]
        url = f"/execute?cmd=ping {random.choice(payloads)}"
        content = f"command=ls {random.choice(payloads)}"
    
    else:  # Default to a generic attack
        url = "/vulnerable?param=attack"
        content = "malicious=data"
    
    return {
        'Method': random.choice(['GET', 'POST']),
        'URL': url,
        'content': content,
        'content-type': random.choice(['application/x-www-form-urlencoded', 'application/json']),
        'Cookie': f'session={random_string(32)}',
        'Length': random.randint(100, 2000),
        'User-Agent': random.choice([
            'Mozilla/5.0 (Windows NT 10.0; Win64; x64)',
            'Mozilla/5.0 (Linux; Android 10)',
            'sqlmap/1.4.7',
            'Nikto/2.1.6',
            'ZAP/2.10.0'
        ])
    }

def generate_synthetic_http_dataset(num_normal=1000, num_attacks=200):
    data = []
    
    # Generate normal requests
    for i in range(num_normal):
        # Create normal HTTP request patterns
        request = {
            'Method': random.choice(['GET', 'POST', 'HEAD', 'PUT']),
            'URL': random.choice([
                f'/index.html',
                f'/products/{random.randint(1, 100)}',
                f'/users/profile/{random_string(8)}',
                f'/search?q={random_string(5)}',
                f'/api/v1/resources/{random.randint(1, 50)}'
            ]),
            'content': '' if random.choice([True, False]) else f'param1={random_string(8)}&param2={random_string(5)}',
            'content-type': random.choice(['application/x-www-form-urlencoded', 'application/json', 'text/plain']),
            'Cookie': f'session={random_string(32)}; pref={random_string(10)}',
            'Length': random.randint(10, 500),
            'User-Agent': random.choice([
                'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36',
                'Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/605.1.15 (KHTML, like Gecko) Version/14.1.1 Safari/605.1.15',
                'Mozilla/5.0 (X11; Linux x86_64; rv:89.0) Gecko/20100101 Firefox/89.0'
            ]),
            'classification': 0  # 0 for normal
        }
        data.append(request)
    
    # Generate attack requests
    attack_types = ['sql_injection', 'xss', 'path_traversal', 'command_injection']
    for i in range(num_attacks):
        # Create attack patterns
        attack_type = random.choice(attack_types)
        request = create_attack_request(attack_type)
        request['classification'] = 1  # 1 for attack
        data.append(request)
    
    # Convert to DataFrame and shuffle
    df = pd.DataFrame(data)
    df = df.sample(frac=1).reset_index(drop=True)
    
    return df

def save_dataset(df, filename="synthetic_http_dataset.csv"):
    """Save the generated dataset to a CSV file"""
    output_dir = "D:\\University\\Software Engineering\\Project\\generated data"
    os.makedirs(output_dir, exist_ok=True)
    output_path = os.path.join(output_dir, filename)
    df.to_csv(output_path, index=False)
    print(f"Dataset saved to: {output_path}")
    return output_path

if __name__ == "__main__":
    # Generate dataset with 2000 normal requests and 500 attack requests
    print("Generating synthetic HTTP dataset...")
    dataset = generate_synthetic_http_dataset(num_normal=2000, num_attacks=500)
    
    # Display dataset statistics
    print(f"Dataset shape: {dataset.shape}")
    print(f"Normal requests: {sum(dataset['classification'] == 0)}")
    print(f"Attack requests: {sum(dataset['classification'] == 1)}")
    
    # Save the dataset
    output_path = save_dataset(dataset)
    
    print("Done! You can now use this dataset to test your model.")