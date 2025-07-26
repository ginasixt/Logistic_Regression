# scripts/main.py
"""
Logistische Regression zur Vorhersage von Diabetes
Dieses Skript lädt einen Datensatz über Gesundheitsindikatoren, 
trainiert ein logistisches Regressionsmodell und bewertet dessen Leistung.
"""
import pandas as pd
import numpy as np
import os
import joblib
import kagglehub

from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    classification_report
)

print(" Lade Datensatz mit kagglehub")
path = kagglehub.dataset_download("alexteboul/diabetes-health-indicators-dataset")

csv_path = os.path.join(path, "diabetes_binary_health_indicators_BRFSS2015.csv")
print(f"[Daten geladen von: {csv_path}")

# Daten lesen
df = pd.read_csv(csv_path)
print(f"Form der Daten: {df.shape}")
print(df.head())

# BEREIGNUNG DER DATEN

# entfernt doppelte Einträge
df.drop_duplicates(inplace=True)

# X sind die Input-Features, y ist die Zielvariable
X = df.drop("Diabetes_binary", axis=1)
y = df["Diabetes_binary"]

# skalliert die Features, damit sie vergleichbar sind --> 
# Age kann riesen Werte haben und β_x wird winzig, 
# nur damit das Gleichgewicht stimmt. --> SKallierung notwendig
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# in Trainings und Testdaten aufteilen 80/20
# test_size=0.2 bedeutet, dass 20% der Daten für den Test verwendet werden
# stratify=y sorgt dafür, dass die Verteilung von y/n Diabetes in Testdaten u Trainingsdaten ca gleich bleiben.
X_train, X_test, y_train, y_test = train_test_split(
    X_scaled, y, test_size=0.2, random_state=42, stratify=y
)

# Training starten
# Kann auch noch class_weight='balanced' 
# sorgt dafür, dass die unterrepräsentierte Klasse (Diabetes = 1) mehr Gewicht im Training bekommt.
# So wird das Modell nicht mehr so stark auf die nicht Diabetiker:innen optimiert.
print("Trainiere Modell")
model = LogisticRegression(max_iter=1000) # maximal 1000 Durchläufe aller Partienten, meist stoppt früher, wenn Konvergenz erreicht ist
model.fit(X_train, y_train)

# Modellbewertung
y_pred = model.predict(X_test)

print("\n=== Modellbewertung ===")

print("Accuracy:", accuracy_score(y_test, y_pred))
# Kein Diabetes: [[TN   FP]
# Ja Diabetes:    [FN   TP]]
print("Confusion Matrix:\n", confusion_matrix(y_test, y_pred))


print("Classification Report:\n", classification_report(y_test, y_pred))

# Feature Gewichtungen
coeff_df = pd.DataFrame({
    "Feature": X.columns,
    "Coefficient": model.coef_[0],
    "OddsRatio": np.exp(model.coef_[0])
}).sort_values(by="OddsRatio", ascending=False)

print("\n=== Einfluss der Features (Odds Ratios) ===")
print(coeff_df)

# Modell und skallierung speichern
os.makedirs("models", exist_ok=True)
joblib.dump(model, "models/logreg_diabetes.pkl")
joblib.dump(scaler, "models/scaler.pkl")
print("[INFO] Modell und Skalierer gespeichert in /models")

