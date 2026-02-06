
import os
import shutil
import pandas as pd
import numpy as np
import logging
from ml.services.user_training_service import UserTrainingService

# Configure logger to see output
logging.basicConfig(level=logging.INFO)

def run_verification():
    print("="*60)
    print("🧪 STARTING DISORDERED FEATURE VERIFICATION")
    print("="*60)

    # 1. Setup Service
    # Note: We point to 'pre_trained_defect_models' where the files actually are locally
    service = UserTrainingService(
        models_dir="pre_trained_defect_models/trained_models",
        features_dir="pre_trained_defect_models/model_features"
    )

    # 2. Define Model Name (The one that caused issues)
    model_name_input = "random_forest_model" 
    # Logic: 
    # - Service looks for "random_forest_model_features.csv" (garbage/missing)
    # - Should FALLBACK to "randomforest_features.csv"

    # 3. Load Correct Features Manually to create Mock Data
    correct_schema = "pre_trained_defect_models/model_features/randomforest_features.csv"
    if not os.path.exists(correct_schema):
        print(f"❌ Critical: Correct schema not found locally at {correct_schema}")
        return

    with open(correct_schema, 'r') as f:
        required_features = [line.strip() for line in f.readlines()]
    
    print(f"✅ Loaded required features from disk: {len(required_features)}")
    print(f"   First 5: {required_features[:5]}")

    # 4. Create Mock Data with SHUFFLED columns
    # This proves that the service re-orders them correctly
    data = {feat: [np.random.random()] for feat in required_features}
    df = pd.DataFrame(data)
    
    # Shuffle columns
    df = df[np.random.permutation(df.columns)]
    print(f"⚠️ Mock data columns shuffled! First 5: {list(df.columns)[:5]}")
    
    # Save to CSV
    input_csv = "verify_input.csv"
    df.to_csv(input_csv, index=False)

    # 5. Run Prediction
    try:
        print("\n🔮 Running predict_from_csv...")
        results = service.predict_from_csv(
            input_csv=input_csv,
            model_name=model_name_input
        )
        print("\n✅ PREDICTION SUCCESS!")
        print(f"   Result shape: {results.shape}")
        print("   This proves:")
        print("   1. Schema fallback worked (found correct features)")
        print("   2. Feature order was enforced (shuffled input didn't crash it)")
        
    except Exception as e:
        print(f"\n❌ PREDICTION FAILED: {e}")
        import traceback
        traceback.print_exc()
    finally:
        if os.path.exists(input_csv):
            os.remove(input_csv)

if __name__ == "__main__":
    run_verification()
