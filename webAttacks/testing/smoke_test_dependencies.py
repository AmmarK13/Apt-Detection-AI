# smoke_test_dependencies.py
import importlib

def test_dependencies():
    print("\n===== Testing Dependencies =====\n")
    required_packages = [
        'pandas', 'numpy' ,'joblib'
    ]
    
    all_installed = True
    for package in required_packages:
        try:
            importlib.import_module(package)
            print(f"✓ {package} is installed")
        except ImportError:
            print(f"✗ {package} is NOT installed")
            all_installed = False
    
    return all_installed

if __name__ == "__main__":
    test_dependencies()