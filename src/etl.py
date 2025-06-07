
import pandas as pd
import os

def load_data(filepath):
    return pd.read_csv(filepath)

def clean_and_transform(df):
    # Drop rows with missing matched_score
    df = df.dropna(subset=['matched_score'])

    # Fill missing values in key text fields
    text_fields = ['skills', 'career_objective', 'responsibilities.1', 'skills_required']
    for field in text_fields:
        if field in df.columns:
            df[field] = df[field].fillna('')

    # Create binary target column: 1 if matched_score >= 0.75 else 0
    df['label'] = df['matched_score'].apply(lambda x: 1 if x >= 0.75 else 0)

    return df

def save_cleaned_data(df, output_path):
    df.to_csv(output_path, index=False)
    print(f"✅ Cleaned data saved to {output_path}")

if __name__ == "__main__":
    raw_path = "sajansshergill/job-success-predictor/data/resume_data.csv"
    processed_path = "sajansshergill/job-success-predictor/data/resume_data.csv"

    # Load, clean, and save
    df = load_data(raw_path)
    df_clean = clean_and_transform(df)
    save_cleaned_data(df_clean, processed_path)
