import pandas as pd
from pathlib import Path

# Paths based on your project structure
BASE_DIR = Path(__file__).resolve().parents[1]
DATA_DIR = BASE_DIR / "data"
OUTPUT_PERFORMANCES = DATA_DIR / "performances_selected_datasets.csv"
OUTPUT_MATRIX = DATA_DIR / "performance_matrix.csv"

def clean_data():
    if OUTPUT_PERFORMANCES.exists():
        df = pd.read_csv(OUTPUT_PERFORMANCES)
        
        # Keep only the lightweight classifiers
        df_clean = df[~df["Classifier"].isin(["SVM", "KNN", "MLP"])]
        
        df_clean.to_csv(OUTPUT_PERFORMANCES, index=False)
        print(f"Success! Removed {len(df) - len(df_clean)} rows containing old SVM, KNN, and MLP data.")
    
    # Delete the matrix so it gets recreated from scratch
    if OUTPUT_MATRIX.exists():
        OUTPUT_MATRIX.unlink()
        print("Deleted performance_matrix.csv. It will be rebuilt automatically.")

if __name__ == "__main__":
    clean_data()