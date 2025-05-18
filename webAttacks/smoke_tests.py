# run_smoke_tests.py
import os
import sys
import time

# Add the project root to the path to import from code module
sys.path.append("d:/University/Software Engineering/Project/Apt-Detection-AI/webAttacks")

# Import individual test modules
from testing.smoke_test_model_loading import test_model_loading
from testing.smoke_test_dependencies import test_dependencies
from testing.smoke_test_data_loading import test_data_loading
from testing.smoke_test_data_compatibility import test_data_compatibility
from testing.smoke_test_prediction import test_prediction
from testing.smoke_test_prediction_values import test_prediction_values
from testing.smoke_test_performance import test_prediction_speed
from testing.smoke_test_model_params import test_model_parameters

def run_all_smoke_tests():
    print("\n===== STARTING SMOKE TESTS FOR WEB ATTACK DETECTION MODEL =====\n")
    
    # Create a results dictionary
    results = {}
    
    # Test 1: Model Loading and Initialization
    print("\n----- Test 1: Model Loading and Initialization -----")
    results["model_loading"] = test_model_loading()
    results["dependencies"] = test_dependencies()
    
    # Test 2: Input Data Validation
    print("\n----- Test 2: Input Data Validation -----")
    sample_data = test_data_loading()
    results["data_loading"] = sample_data is not None
    results["data_compatibility"] = test_data_compatibility()
    
    # Test 3: Basic Prediction Functionality
    print("\n----- Test 3: Basic Prediction Functionality -----")
    results["prediction"] = test_prediction()
    results["prediction_values"] = test_prediction_values()
    
    # Test 4: Performance Testing
    print("\n----- Test 4: Performance Testing -----")
    results["prediction_speed"] = test_prediction_speed()
    results["model_parameters"] = test_model_parameters()
    
    # Print summary
    print("\n===== SMOKE TEST SUMMARY =====\n")
    for test_name, passed in results.items():
        status = "✓ PASSED" if passed else "✗ FAILED"
        print(f"{status}: {test_name}")
    
    # Overall result
    all_passed = all(results.values())
    print(f"\nOverall smoke test result: {'✓ PASSED' if all_passed else '✗ FAILED'}")
    
    return all_passed

if __name__ == "__main__":
    run_all_smoke_tests()