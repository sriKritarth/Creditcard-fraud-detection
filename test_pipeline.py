"""
Test script to verify the complete data processing and model training pipeline.
"""

import os
import sys
import subprocess
from pathlib import Path


def run_command(cmd, description):
    """Run a command and return success status."""
    print(f"\n🔄 {description}")
    print(f"Running: {cmd}")
    try:
        result = subprocess.run(cmd, shell=True, check=True, capture_output=True, text=True)
        print(f"✅ {description} - SUCCESS")
        if result.stdout:
            print(f"Output: {result.stdout.strip()}")
        return True
    except subprocess.CalledProcessError as e:
        print(f"❌ {description} - FAILED")
        print(f"Error: {e.stderr.strip()}")
        return False


def check_file_exists(filepath, description):
    """Check if a file exists and print its size."""
    if os.path.exists(filepath):
        size = os.path.getsize(filepath)
        print(f"✅ {description} - EXISTS ({size:,} bytes)")
        return True
    else:
        print(f"❌ {description} - NOT FOUND")
        return False


def main():
    """Test the complete pipeline."""
    print("🚀 Testing Credit Card Fraud Detection Pipeline")
    print("=" * 50)
    
    # Check if raw data exists
    raw_data = "data/raw/creditcard.csv"
    if not check_file_exists(raw_data, "Raw data file"):
        print("❌ Raw data not found. Please ensure data/raw/creditcard.csv exists.")
        return False
    
    # Test data processing
    if not run_command(
        "python src/data/make_dataset.py data/raw/creditcard.csv",
        "Data processing"
    ):
        return False
    
    # Check processed data
    train_data = "data/processed/train.csv"
    test_data = "data/processed/test.csv"
    
    if not check_file_exists(train_data, "Train data"):
        return False
    if not check_file_exists(test_data, "Test data"):
        return False
    
    # Test model training
    if not run_command(
        "python src/models/train_model.py data/processed/",
        "Model training"
    ):
        return False
    
    # Check model file
    model_file = "models/model.joblib"
    if not check_file_exists(model_file, "Trained model"):
        return False
    
    # Test model loading
    if not run_command(
        "python -c \"from joblib import load; model = load('models/model.joblib'); print('Model type:', type(model))\"",
        "Model loading test"
    ):
        return False
    
    print("\n🎉 All tests passed! Pipeline is working correctly.")
    return True


if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)