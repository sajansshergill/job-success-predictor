import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier
from sklearn.metrics import classification_report, accuracy_score
import os
import joblib

def load_data(X_path, y_path):
    X = pd.read_csv(X_path)
    y = pd.read_csv(y_path).squeeze()
    return X, y

def log_and_print(f, message):
    print(message)
    f.write(message + "\n")

def train_and_evaluate_models(X, y, output_path):
    os.makedirs(os.path.dirname(output_path), exist_ok=True)

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    models = {
        "Logistic Regression": LogisticRegression(max_iter=1000),
        "Random Forest": RandomForestClassifier(n_estimators=100),
        "XGBoost": XGBClassifier(use_label_encoder=False, eval_metric='logloss')
    }

    best_model = None
    best_score = 0
    best_model_name = ""

    with open(output_path, 'w') as f:
        for name, model in models.items():
            log_and_print(f, f"\n🚀 Model: {name}")
            model.fit(X_train, y_train)
            y_pred = model.predict(X_test)

            acc = accuracy_score(y_test, y_pred)
            report = classification_report(y_test, y_pred)

            log_and_print(f, f"✅ Accuracy: {acc:.4f}")
            log_and_print(f, "📊 Classification Report:\n" + report)

            if acc > best_score:
                best_score = acc
                best_model = model
                best_model_name = name

        # Save best model
        os.makedirs("models", exist_ok=True)
        joblib.dump(best_model, "models/best_model.pkl")
        log_and_print(f, f"\n💾 Best model ({best_model_name}) saved to models/best_model.pkl")

if __name__ == "__main__":
    X_path = "data/processed/X_features.csv"
    y_path = "data/processed/y_target.csv"
    output_path = "outputs/model_evaluation.txt"

    X, y = load_data(X_path, y_path)
    train_and_evaluate_models(X, y, output_path)
