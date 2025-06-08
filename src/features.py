import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import os

def load_clean_data(filepath):
    return pd.read_csv(filepath)

def create_features(df):
    # Make sure 'skills' and 'skills_required' are strings
    df['skills'] = df['skills'].fillna('').astype(str)
    df['skills_required'] = df['skills_required'].fillna('').astype(str)

    # TF-IDF on 'skills'
    tfidf = TfidfVectorizer(max_features=100)
    tfidf_matrix = tfidf.fit_transform(df['skills'])
    tfidf_df = pd.DataFrame(tfidf_matrix.toarray(), columns=tfidf.get_feature_names_out())

    # Cosine similarity between 'skills' and 'skills_required'
    combined = df['skills'] + ' ' + df['skills_required']
    combined_tfidf = TfidfVectorizer().fit_transform(combined)
    similarity = cosine_similarity(combined_tfidf)
    skill_similarity_scores = similarity.diagonal()
    df['skills_similarity'] = skill_similarity_scores

    # Combine features
    features = pd.concat([tfidf_df, df[['skills_similarity']]], axis=1)
    target = df['label']

    return features, target

def save_features(X, y, output_dir):
    os.makedirs(output_dir, exist_ok=True)
    X.to_csv(os.path.join(output_dir, "X_features.csv"), index=False)
    y.to_csv(os.path.join(output_dir, "y_target.csv"), index=False)
    print(f"✅ Features and target saved to {output_dir}")

if __name__ == "__main__":
    input_path = "data/processed/cleaned_resume_data.csv"
    output_dir = "data/processed"

    df = load_clean_data(input_path)
    X, y = create_features(df)
    save_features(X, y, output_dir)
