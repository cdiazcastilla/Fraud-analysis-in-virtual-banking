from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from xgboost import XGBClassifier
from sklearn.model_selection import train_test_split, StratifiedKFold, cross_val_score, GridSearchCV
from imblearn.over_sampling import SMOTE
from imblearn.pipeline import Pipeline as ImbPipeline
import pandas as pd
import numpy as np

# Load data
df = pd.read_csv('creditcard_synthetic.csv')
X = df.drop(columns=['Class'])
y = df['Class']

# Split
X_train, X_test, y_train, y_test = train_test_split(X, y, stratify=y, test_size=0.2, random_state=42)

# Define pipeline steps
pipeline = ImbPipeline([
    ('scaler', StandardScaler()),
    ('smote', SMOTE(random_state=42)),
    ('model', XGBClassifier(use_label_encoder=False, eval_metric='logloss', random_state=42))
])

# Cross-validation strategy
cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

# Grid Search parameters
params = {
    'model__n_estimators': [100, 200],
    'model__max_depth': [4, 6],
    'model__learning_rate': [0.05, 0.1],
}

# Grid search
grid = GridSearchCV(pipeline, param_grid=params, cv=cv, scoring='f1', verbose=1, n_jobs=-1)
grid.fit(X_train, y_train)

# Best model
print(f"🔥 Best params: {grid.best_params_}")
print(f"🎯 Best CV F1-score: {grid.best_score_:.4f}")

# Evaluate on test set
from sklearn.metrics import classification_report, confusion_matrix
y_pred = grid.predict(X_test)

print("\n🧠 Test set classification report:")
print(classification_report(y_test, y_pred))

# Confusion Matrix
import matplotlib.pyplot as plt
import seaborn as sns

cm = confusion_matrix(y_test, y_pred)
sns.heatmap(cm, annot=True, fmt='d', cmap='Purples', xticklabels=['Legit', 'Fraud'], yticklabels=['Legit', 'Fraud'])
plt.xlabel('Predicted')
plt.ylabel('Actual')
plt.title('Confusion Matrix - Final Tuned Model')
plt.show()
