import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestRegressor
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.pipeline import make_pipeline
import pickle
import os

# Create dummy data for demonstration
data = {
    'essay': [
        "This is a great essay with good structure and vocabulary.",
        "Bad essay. Short. Poor grammar.",
        "Average essay, some good points but needs improvement.",
        "Excellent work! Very detailed and well-argued.",
        "Terrible. Completely off-topic and nonsensical.",
        "Good effort, but lacks depth in analysis.",
        "A bit repetitive, but clearly written.",
        "Outstanding! One of the best essays I've read.",
        "Needs a lot of work. Spelling mistakes everywhere.",
        "Decent essay with a clear introduction and conclusion."
    ],
    'score': [8, 3, 6, 9, 2, 7, 6, 10, 4, 7]
}

df = pd.DataFrame(data)

# Create a pipeline that converts text to features and then regresses
# We use TF-IDF for simple text representation
model = make_pipeline(TfidfVectorizer(), RandomForestRegressor(n_estimators=100, random_state=42))

# Train the model
print("Training model...")
model.fit(df['essay'], df['score'])

# Save the model
model_path = os.path.join(os.path.dirname(__file__), 'essay_scorer_model.pkl')
with open(model_path, 'wb') as f:
    pickle.dump(model, f)

print(f"Model saved to {model_path}")

# Test the model
test_essay = "This is a good essay."
predicted_score = model.predict([test_essay])[0]
print(f"Test prediction for '{test_essay}': {predicted_score:.2f}")
