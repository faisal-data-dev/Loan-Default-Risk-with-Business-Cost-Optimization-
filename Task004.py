# task_4_loan_default.py

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import confusion_matrix, classification_report

np.random.seed(42)

# 1. Generate Synthetic Home Credit Default Risk Dataset[cite: 1]
print("Cleaning and preprocessing the dataset...")
n_samples = 2000
data = pd.DataFrame({
    'AMT_CREDIT': np.random.uniform(20000, 500000, size=n_samples),
    'AMT_INCOME_TOTAL': np.random.uniform(10000, 150000, size=n_samples),
    'EXT_SOURCE_3': np.random.uniform(0.1, 0.9, size=n_samples),
    'DAYS_EMPLOYED': np.random.randint(-10000, 0, size=n_samples),
    'TARGET': np.random.choice([0, 1], size=n_samples, p=[0.90, 0.10])
})

# Preprocessing[cite: 1]
X = data.drop(columns=['TARGET'])
y = data['TARGET']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, stratify=y, random_state=42)

scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# 2. Train Binary Classification Model[cite: 1]
print("Training Logistic Regression model...")
clf = LogisticRegression(class_weight='balanced', random_state=42)
clf.fit(X_train_scaled, y_train)
y_probs = clf.predict_proba(X_test_scaled)[:, 1]

# 3. Define Business Costs[cite: 1]
# False Negative (predicting non-default when they actually default) -> High risk/loss
cost_fn = 10000 
# False Positive (predicting default when they are a good customer) -> Opportunity cost
cost_fp = 1000  

best_threshold = 0.5
min_cost = float('inf')
optimal_cm = None

# 4. Adjust the model threshold to minimize total business cost[cite: 1]
print("Optimizing decision threshold based on cost-benefit analysis...")
for th in np.linspace(0.1, 0.9, 81):
    preds = (y_probs >= th).astype(int)
    cm = confusion_matrix(y_test, preds)
    
    if cm.size == 4:
        tn, fp, fn, tp = cm.ravel()
        total_cost = (fn * cost_fn) + (fp * cost_fp)
        
        if total_cost < min_cost:
            min_cost = total_cost
            best_threshold = th
            optimal_cm = cm

print("\n--- Cost Optimization Results ---")
print(f"Default Threshold (0.50) Standard Evaluation:")
print(classification_report(y_test, (y_probs >= 0.5).astype(int)))
print(f"\nOptimal Decision Threshold Found: {best_threshold:.2f}")
print(f"Minimized Total Business Cost: ${min_cost:,}")
print(f"Optimized Confusion Matrix:\n{optimal_cm}")