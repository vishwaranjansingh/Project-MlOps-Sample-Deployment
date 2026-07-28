import os
import joblib
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.ensemble import RandomForestRegressor
from sklearn.pipeline import Pipeline
from sklearn.metrics import mean_squared_error, r2_score

def generate_synthetic_data(n_samples=1000, random_state=42):
    np.random.seed(random_state)
    
    # 10th and 12th percentage scores (range 50 to 98)
    score_10th = np.random.uniform(50.0, 98.0, n_samples)
    score_12th = np.random.uniform(50.0, 98.0, n_samples)
    
    education_levels = ['High School', 'Bachelor', 'Master', 'PhD']
    education = np.random.choice(education_levels, size=n_samples, p=[0.1, 0.5, 0.3, 0.1])
    
    jobs = ['Software Engineer', 'Data Scientist', 'Manager', 'Business Analyst', 'Executive', 'Sales Specialist']
    current_job = np.random.choice(jobs, size=n_samples)
    
    years_exp = np.random.uniform(0.5, 25.0, n_samples)
    
    # Formula to simulate target: Mind Readiness / Career Aptitude Alignment Score (0 to 100)
    edu_weight = {'High School': 5, 'Bachelor': 15, 'Master': 25, 'PhD': 35}
    job_weight = {
        'Software Engineer': 20,
        'Data Scientist': 22,
        'Manager': 18,
        'Business Analyst': 16,
        'Executive': 25,
        'Sales Specialist': 14
    }
    
    base_score = (
        0.25 * score_10th +
        0.35 * score_12th +
        np.vectorize(edu_weight.get)(education) +
        np.vectorize(job_weight.get)(current_job) +
        0.8 * years_exp +
        np.random.normal(0, 3, n_samples) # noise
    )
    
    # Normalize target score to 0 - 100 range
    readiness_score = np.clip((base_score / base_score.max()) * 100, 0, 100)
    
    df = pd.DataFrame({
        'score_10th': score_10th,
        'score_12th': score_12th,
        'education_level': education,
        'current_job': current_job,
        'years_of_experience': years_exp,
        'readiness_score': readiness_score
    })
    
    return df

def train_and_save_model(output_path='models/model.pkl'):
    print("Generating synthetic data for Human Mind & Aptitude evaluation model...")
    df = generate_synthetic_data()
    
    X = df.drop(columns=['readiness_score'])
    y = df['readiness_score']
    
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    num_features = ['score_10th', 'score_12th', 'years_of_experience']
    cat_features = ['education_level', 'current_job']
    
    preprocessor = ColumnTransformer(
        transformers=[
            ('num', StandardScaler(), num_features),
            ('cat', OneHotEncoder(handle_unknown='ignore', sparse_output=False), cat_features)
        ]
    )
    
    model_pipeline = Pipeline(steps=[
        ('preprocessor', preprocessor),
        ('regressor', RandomForestRegressor(n_estimators=100, random_state=42))
    ])
    
    print("Training model pipeline...")
    model_pipeline.fit(X_train, y_train)
    
    y_pred = model_pipeline.predict(X_test)
    mse = mean_squared_error(y_test, y_pred)
    r2 = r2_score(y_test, y_pred)
    
    print(f"Model Evaluation Results:")
    print(f" - Mean Squared Error (MSE): {mse:.4f}")
    print(f" - R^2 Score: {r2:.4f}")
    
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    joblib.dump(model_pipeline, output_path)
    print(f"Successfully saved trained model pipeline to '{output_path}'")

if __name__ == '__main__':
    train_and_save_model()
