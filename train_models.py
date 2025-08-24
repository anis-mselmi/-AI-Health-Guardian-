import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.linear_model import LogisticRegression
import pickle

# ----- Symptoms Chat -----
data = {
    "symptoms": [
        "fever, cough, shortness of breath",
        "thirst, frequent urination, fatigue",
        "chest pain, shortness of breath, dizziness",
        "headache, nausea, sensitivity to light"
    ],
    "disease": ["COVID-19", "Diabetes", "Heart Disease", "Migraine"],
    "advice": [
        "Get tested, stay hydrated, rest",
        "Check blood sugar, follow diet, exercise",
        "See a cardiologist, avoid heavy exertion",
        "Rest, take pain relief, avoid bright lights"
    ]
}

df = pd.DataFrame(data)
vectorizer = CountVectorizer()
X = vectorizer.fit_transform(df["symptoms"])
y = df["disease"]

chat_model = LogisticRegression()
chat_model.fit(X, y)

# حفظ النموذج
with open("models/chat_model.pkl", "wb") as f:
    pickle.dump((chat_model, vectorizer, df), f)

print("✅ Symptoms Chat model saved!")
