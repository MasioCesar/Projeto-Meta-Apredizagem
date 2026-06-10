import pandas as pd
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import StratifiedKFold, cross_validate
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.neural_network import MLPClassifier
from xgboost import XGBClassifier

# Import your existing pipeline functions
from common import load_data, build_model, OUTPUT_DIR

def run_algorithm_variance_test():
    X, y, numeric_cols, classifier_cols = load_data()
    encoder = LabelEncoder()
    y_encoded = encoder.fit_transform(y)
    n_splits = min(5, pd.Series(y_encoded).value_counts().min())
    cv = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=42)

    # 1. Define the Baseline vs. Rank 1 feature strategies
    feature_sets = {
        "Baseline_Pure_Stats": {
            "semantic_mode": "none", 
            "domain_mode": "none"
        },
        "Rank_1_Contextual": {
            "semantic_mode": "tags_fixed_vocab", 
            "domain_mode": "label_score_interaction"
        }
    }

    # 2. Define the diverse algorithms to test
    classifiers = {
        "RandomForest": RandomForestClassifier(n_estimators=300, random_state=42, class_weight="balanced"),
        "XGBoost": XGBClassifier(random_state=42, eval_metric="mlogloss"),
        "SVM (Linear)": SVC(kernel="linear", class_weight="balanced", random_state=42),
        "MLP (Neural Net)": MLPClassifier(hidden_layer_sizes=(100, 50), max_iter=500, random_state=42)
    }

    results_list = []

    for feature_name, params in feature_sets.items():
        print(f"\n[{feature_name.upper()}]")
        
        for clf_name, clf in classifiers.items():
            print(f"Training {clf_name}...")
            
            # Build your existing pipeline
            pipeline = build_model(
                numeric_cols, 
                semantic_mode=params["semantic_mode"], 
                domain_mode=params["domain_mode"]
            )
            
            # Swap the final step of the pipeline with the new algorithm
            pipeline.steps[-1] = ("model", clf)
            
            # n_jobs=-1 will utilize all your CPU threads to handle SVM/MLP training faster
            results = cross_validate(
                pipeline,
                X,
                y_encoded,
                cv=cv,
                scoring={"accuracy": "accuracy", "f1_macro": "f1_macro"},
                return_train_score=False,
                n_jobs=-1 
            )
            
            f1_macro = results["test_f1_macro"].mean()
            acc = results["test_accuracy"].mean()
            
            results_list.append({
                "Feature_Set": feature_name,
                "Algorithm": clf_name,
                "Accuracy": acc,
                "F1-Macro": f1_macro
            })
            print(f"  -> F1-Macro: {f1_macro:.4f} | Accuracy: {acc:.4f}")

    # Save to a clean CSV for your paper
    df_results = pd.DataFrame(results_list)
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    output_path = OUTPUT_DIR / "algorithm_agnostic_test.csv"
    df_results.to_csv(output_path, index=False)
    print(f"\nSaved matrix to: {output_path}")

if __name__ == "__main__":
    run_algorithm_variance_test()