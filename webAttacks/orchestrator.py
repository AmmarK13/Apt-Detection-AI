import sys
from smoke_tests import run_all_smoke_tests
from main import run_data_pipeline

def main():
    print("Starting Smoke Tests...")
    try:
        smoke_result = run_all_smoke_tests()
    except Exception as e:
        print(f"Smoke tests raised an exception: {e}")
        sys.exit(1)

    if not smoke_result:
        print("Smoke tests failed! Aborting data pipeline run.")
        sys.exit(1)

    print("Smoke tests passed. Proceeding with data pipeline...")
    try:
        run_data_pipeline()
    except Exception as e:
        print(f"Data pipeline failed: {e}")
        sys.exit(1)

    print("Data pipeline completed successfully.")

if __name__ == "__main__":
    main()
