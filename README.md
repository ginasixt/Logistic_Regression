# Diabetes Prediction with Logistic Regression

This project uses the "Diabetes Health Indicators" dataset from Kaggle to train and evaluate a logistic regression model that predicts whether a person has diabetes based on health-related features (e.g., BMI, age, blood pressure, etc.).

---

## Project Summary

- Language: Python
- Libraries: `scikit-learn`, `pandas`, `numpy`, `kagglehub`, `joblib`
- Model: Logistic Regression (from `sklearn`)
- Dataset: [Diabetes Health Indicators - BRFSS 2015](https://www.kaggle.com/datasets/alexteboul/diabetes-health-indicators-dataset)
- Goal: Predict binary outcome — **diabetes: yes (1) or no (0)**

---

## Workflow

1.  **Download the dataset** using `kagglehub`
2.  **Preprocess data**: remove duplicates, scale features
3.  **Split dataset**: 80% for training, 20% for testing
4.  **Train logistic regression** model on training set
5.  **Evaluate** model using metrics like:
   - Accuracy
   - Confusion Matrix
   - Precision / Recall / F1-Score
6.  **Analyze feature impact** via coefficients & odds ratios
7.  **Save the model & scaler** using `joblib`

---

##  Key Results (Before Class Balancing)

```text
Accuracy: 0.8505283799978212
Confusion Matrix:
 [[37956   920]
 [ 5940  1079]]
```
 Accuracy: How many predictions where generally correct? \
 N Diabetes: [[TN   FP]  \
 Y Diabetes:  [FN   TP]]
```text
Classification Report:
               precision    recall  f1-score   support

         0.0       0.86      0.98      0.92     38876
         1.0       0.54      0.15      0.24      7019


    accuracy                           0.85     45895
   macro avg       0.70      0.57      0.58     45895
weighted avg       0.81      0.85      0.81     45895
```
Therefore the model has high accuracy (86% for non-diabetics), but low sensitivity (15% for diabetics). 

- **precision:** Proportion of positively predicted cases that are actually positive \
Of the predicted “diabetics” → how many are actually ill? 
- **recall:** Proportion of actually positive cases that were correctly detected (sensitivity) \
Of the real diabetics → how many were detected?
- **f1-score:** Harmony between precision and recall: “How balanced is the performance?” 
- **support:** &nbsp; Number of real cases of this class in the test set 
- **Macro Avg:** &nbsp; Average of classes 0 and 1 – equally weighted 
- **Weighted Av:** &nbsp; Average, weighted according to class occurrence 

```text
=== Einfluss der Features (Odds Ratios) ===
                 Feature  Coefficient  OddsRatio
13               GenHlth     0.532904   1.703873
3                    BMI     0.397746   1.488466
18                   Age     0.373564   1.452904
0                 HighBP     0.366516   1.442700
1               HighChol     0.279209   1.322084
2              CholCheck     0.252001   1.286597
17                   Sex     0.130053   1.138889
6   HeartDiseaseorAttack     0.065459   1.067649
16              DiffWalk     0.051013   1.052337
5                 Stroke     0.026453   1.026806
11         AnyHealthcare     0.021369   1.021599
9                Veggies    -0.004009   0.995999
12           NoDocbcCost    -0.005119   0.994894
8                 Fruits    -0.012901   0.987182
4                 Smoker    -0.013218   0.986869
7           PhysActivity    -0.013637   0.986456
19             Education    -0.019621   0.980570
14              MentHlth    -0.033346   0.967204
15              PhysHlth    -0.060324   0.941460
20                Income    -0.092928   0.911259
10     HvyAlcoholConsump    -0.190441   0.826595
```




