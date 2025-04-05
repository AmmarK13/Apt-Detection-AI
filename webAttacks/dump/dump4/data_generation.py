import pandas as pd
import random
import string
import os
import re

def random_string(length):
    """Generate a random string of specified length"""
    return ''.join(random.choice(string.ascii_letters + string.digits) for _ in range(length))

def create_attack_request(attack_type):
    """Create different types of attack requests similar to CSIC dataset"""
    if attack_type == 'sql_injection':
        payloads = [
            "' OR 1=1 --",
            "'; DROP TABLE users; --",
            "' UNION SELECT username, password FROM users --",
            "1' OR '1'='1",
            "admin' --",
            "1; SELECT * FROM information_schema.tables"
        ]
        url = f"/tienda1/publico/anadir.jsp?id={random.choice(payloads)}"
        content = f"username={random.choice(payloads)}&password=test123"
        
    elif attack_type == 'xss':
        payloads = [
            "<script>alert('XSS')</script>",
            "<img src='x' onerror='alert(1)'>",
            "javascript:alert(document.cookie)",
            "<svg onload='alert(1)'>",
            "'\"><script>alert(1)</script>"
        ]
        url = f"/tienda1/publico/registro.jsp?nombre={random.choice(payloads)}"
        content = f"nombre={random.choice(payloads)}&apellidos=test&email=test@example.com"
        
    elif attack_type == 'path_traversal':
        payloads = [
            "../../../etc/passwd",
            "..\\..\\..\\windows\\system32\\config\\sam",
            "file:///etc/passwd",
            "/var/www/html/config.php",
            "../../.env"
        ]
        url = f"/tienda1/publico/img.jsp?file={random.choice(payloads)}"
        content = ""
        
    elif attack_type == 'command_injection':
        payloads = [
            "| cat /etc/passwd",
            "; rm -rf /",
            "& net user",
            "|| whoami",
            "`ping -c 4 attacker.com`"
        ]
        url = f"/tienda1/publico/exec.jsp?cmd=ping {random.choice(payloads)}"
        content = f"command=ls {random.choice(payloads)}"
    
    else:  # Default to a generic attack
        url = "/tienda1/publico/login.jsp?username=admin'--"
        content = "username=admin'--&password=anything"
    
    return {
        'Method': random.choice(['GET', 'POST']),
        'URL': url,
        'content': content,
        'content-type': 'application/x-www-form-urlencoded',
        'Accept': 'text/html,application/xhtml+xml,application/xml',
        'cookie': f'JSESSIONID={random_string(32)}',
        'classification': 1  # 1 for attack
    }

def generate_synthetic_http_dataset(num_normal=10000, num_attacks=2080):
    data = []
    
    # Generate normal requests based on CSIC patterns
    normal_urls = [
        '/tienda1/publico/index.jsp',
        '/tienda1/publico/login.jsp',
        '/tienda1/publico/registro.jsp',
        '/tienda1/publico/productos.jsp',
        '/tienda1/publico/detalles.jsp',
        '/tienda1/publico/micarrito.jsp',
        '/tienda1/publico/pagar.jsp',
        '/tienda1/publico/logout.jsp'
    ]
    
    normal_params = [
        {'nombre': 'John', 'apellidos': 'Doe', 'email': 'john@example.com'},
        {'username': 'user123', 'password': 'pass123'},
        {'id': '42', 'cantidad': '2'},
        {'tarjeta': '4111111111111111', 'cvv': '123', 'fecha': '12/25'},
        {'buscar': 'producto', 'categoria': 'electronica'}
    ]
    
    for i in range(num_normal):
        # Create normal HTTP request patterns
        url = random.choice(normal_urls)
        params = random.choice(normal_params)
        
        # Add query parameters to some URLs
        if random.random() < 0.3 and '?' not in url:
            param_key = random.choice(list(params.keys()))
            url += f'?{param_key}={params[param_key]}'
            
        # Create content for POST requests
        if random.random() < 0.5:  # 50% POST, 50% GET
            method = 'POST'
            content = '&'.join([f"{k}={v}" for k, v in params.items()])
        else:
            method = 'GET'
            content = ''
            
        request = {
            'Method': method,
            'URL': url,
            'content': content,
            'content-type': 'application/x-www-form-urlencoded',
            'Accept': 'text/html,application/xhtml+xml,application/xml',
            'cookie': f'JSESSIONID={random_string(32)}',
            'classification': 0  # 0 for normal
        }
        data.append(request)
    
    # Generate attack requests
    attack_types = ['sql_injection', 'xss', 'path_traversal', 'command_injection']
    for i in range(num_attacks):
        # Create attack patterns
        attack_type = random.choice(attack_types)
        request = create_attack_request(attack_type)
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
    dataset = generate_synthetic_http_dataset(num_normal=10000, num_attacks=810)
    
    # Display dataset statistics
    print(f"Dataset shape: {dataset.shape}")
    print(f"Normal requests: {sum(dataset['classification'] == 0)}")
    print(f"Attack requests: {sum(dataset['classification'] == 1)}")
    
    # Save the dataset
    output_path = save_dataset(dataset)
    
    print("Done! You can now use this dataset to test your model.")