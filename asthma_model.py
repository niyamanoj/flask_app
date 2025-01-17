# Import Libraries
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from imblearn.over_sampling import SMOTE # type: ignore
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
from sklearn.multiclass import OneVsRestClassifier
from sklearn.metrics import ConfusionMatrixDisplay
import warnings
warnings.filterwarnings('ignore')

# Load Dataset
df = pd.read_csv("C:/Users/manoj/OneDrive/Desktop/ai/balanced_asthma_data.csv")
# Outlier Detection and Handling
for column in df.columns:
    if column != 'Condition':
        Q1 = np.quantile(df[column], 0.25)
        Q3 = np.quantile(df[column], 0.75)
        IQR = Q3 - Q1
        lower_bound = Q1 - 1.5 * IQR
        upper_bound = Q3 + 1.5 * IQR
        df[column] = np.clip(df[column], lower_bound, upper_bound)

# Feature Separation
X = df.drop('Condition', axis=1)
y = df['Condition']

# Train-Test Split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

# Handle Class Imbalance with SMOTE
smote = SMOTE(random_state=42)
X_train_smote, y_train_smote = smote.fit_resample(X_train, y_train)

# Feature Scaling
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train_smote)
X_test_scaled = scaler.transform(X_test)

# Models
models = {
    'Logistic Regression': LogisticRegression(multi_class='multinomial', solver='lbfgs'),
    'KNN': KNeighborsClassifier(),
    'SVM': SVC(decision_function_shape='ovr', probability=True)
}
# Train and Evaluate Models
for model_name, model in models.items():
    model.fit(X_train_scaled, y_train_smote)
    y_pred = model.predict(X_test_scaled)
    print(f"Model: {model_name}")
    print("Accuracy:", accuracy_score(y_test, y_pred))
    print("Classification Report:\n", classification_report(y_test, y_pred))

    # Confusion Matrix
    cm = confusion_matrix(y_test, y_pred)
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=np.unique(y))
    disp.plot(cmap='Blues')
    plt.title(f"Confusion Matrix for {model_name}")
    plt.show()

# Optional: ROC-AUC Curve
from sklearn.metrics import roc_curve, roc_auc_score

plt.figure(figsize=(8, 6))
for model_name, model in models.items():
    if hasattr(model, "predict_proba"):
        y_pred_proba = model.predict_proba(X_test_scaled)
        fpr, tpr, _ = roc_curve(y_test, y_pred_proba[:, 1], pos_label=np.unique(y)[1])
        auc_score = roc_auc_score(y_test, y_pred_proba, multi_class="ovr")
        plt.plot(fpr, tpr, label=f"{model_name} (AUC = {auc_score:.2f})")
plt.plot([0, 1], [0, 1], 'k--', label="Random Classifier")
plt.title("ROC-AUC Curve")
plt.xlabel("False Positive Rate")
plt.ylabel("True Positive Rate")
plt.legend()
plt.show()

svm_model_asthma = SVC(probability=True)
svm_model_asthma.fit(X_train_scaled,y_train_smote)
import joblib
import pickle
# Save the trained model (SVM model in this case)
joblib.dump(svm_model_asthma, 'asthma_model.pkl')

# Save the scaler for future use
joblib.dump(scaler, 'scaler2.pkl')


