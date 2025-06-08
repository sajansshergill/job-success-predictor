import pandas as pd
import os

def load_data(filepath):
    return pd.read_csv(filepath)

def clean_and_transform(df):
    df = df.dropna(subset=['matched_score'])

    text_fields = ['skills', 'career_objective', 'responsibilities.1', 'skills_required']
    for field in text_fields:
        if field in df.columns:
            df[field] = df[field].fillna('')

    df['label'] = df['matched_score'].apply(lambda x: 1 if x >= 0.75 else 0)
    return df

def save_cleaned_data(df, output_dir, filename):
    os.makedirs(output_dir, exist_ok=True)
    output_path = os.path.join(output_dir, filename)
    df.to_csv(output_path, index=False)
    print(f"✅ Cleaned data saved to {output_path}")

if __name__ == "__main__":
    raw_path = "data/raw/resume_data.csv"                        # Relative path
    output_dir = "data/processed"                            # Relative path
    output_file = "cleaned_resume_data.csv"

    df = load_data(raw_path)
    df_clean = clean_and_transform(df)
    save_cleaned_data(df_clean, output_dir, output_file)
