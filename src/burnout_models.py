"""
Employee burnout prediction: five classifiers, PCA visualization, and K-Means profiling.

Run from the project root:
    python src/burnout_models.py
"""

# Core libraries
import os
import pandas as pd
import numpy as np

# Visualization
import matplotlib.pyplot as plt
import seaborn as sns

# ML tools
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.metrics import (
    classification_report,
    accuracy_score,
    confusion_matrix,
    ConfusionMatrixDisplay,
)
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans

RANDOM_STATE = 42
DATA_PATH = "data/tech_mental_health_burnout.csv"
OUTPUT_DIR = "outputs"

# Create the output folder if it's missing (GitHub doesn't store empty folders)
os.makedirs(OUTPUT_DIR, exist_ok=True)

# ---------------------------------------------------------------------------
# 1. Load data
# ---------------------------------------------------------------------------
df = pd.read_csv(DATA_PATH)

print(df.head())
print(df.info())
print(df.describe())
print(df.isnull().sum())
print(df["burnout_level"].value_counts())

# ---------------------------------------------------------------------------
# 2. Preprocessing
# ---------------------------------------------------------------------------
# Fill missing values: median for numeric, mode for categorical
num_cols = df.select_dtypes(include=["int64", "float64"]).columns
df[num_cols] = df[num_cols].fillna(df[num_cols].median())

cat_cols = df.select_dtypes(include=["object"]).columns
df[cat_cols] = df[cat_cols].fillna(df[cat_cols].mode().iloc[0])

# Label-encode categorical columns (one encoder per column)
for col in cat_cols:
    df[col] = LabelEncoder().fit_transform(df[col])

# Features and target. burnout_score is dropped to prevent data leakage.
X = df.drop(["burnout_level", "burnout_score"], axis=1)
y = df["burnout_level"]

print("Columns used in X:")
print(X.columns)

# Stratified 80/20 split so every burnout level appears in both sets
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=RANDOM_STATE, stratify=y
)

# Scaled features for distance- and gradient-based models
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# ---------------------------------------------------------------------------
# 3. Classification models
# ---------------------------------------------------------------------------
# Logistic Regression (scaled)
model_lr = LogisticRegression(max_iter=1000)
model_lr.fit(X_train_scaled, y_train)
y_pred_lr = model_lr.predict(X_test_scaled)

print("\nLogistic Regression")
print("Accuracy:", accuracy_score(y_test, y_pred_lr))
print(classification_report(y_test, y_pred_lr, zero_division=0))

# Decision Tree (raw features, depth-limited)
model_dt = DecisionTreeClassifier(max_depth=5, random_state=RANDOM_STATE)
model_dt.fit(X_train, y_train)
y_pred_dt = model_dt.predict(X_test)

print("\nDecision Tree")
print("Accuracy:", accuracy_score(y_test, y_pred_dt))
print(classification_report(y_test, y_pred_dt, zero_division=0))

# Random Forest (raw features)
model_rf = RandomForestClassifier(n_estimators=100, random_state=RANDOM_STATE)
model_rf.fit(X_train, y_train)
y_pred_rf = model_rf.predict(X_test)

print("\nRandom Forest")
print("Accuracy:", accuracy_score(y_test, y_pred_rf))
print(classification_report(y_test, y_pred_rf, zero_division=0))

# KNN (scaled)
model_knn = KNeighborsClassifier(n_neighbors=5)
model_knn.fit(X_train_scaled, y_train)
y_pred_knn = model_knn.predict(X_test_scaled)

print("\nKNN")
print("Accuracy:", accuracy_score(y_test, y_pred_knn))
print(classification_report(y_test, y_pred_knn, zero_division=0))

# Naive Bayes (scaled)
model_nb = GaussianNB()
model_nb.fit(X_train_scaled, y_train)
y_pred_nb = model_nb.predict(X_test_scaled)

print("\nNaive Bayes")
print("Accuracy:", accuracy_score(y_test, y_pred_nb))
print(classification_report(y_test, y_pred_nb, zero_division=0))

# ---------------------------------------------------------------------------
# 4. Results: feature importance and model comparison
# ---------------------------------------------------------------------------
importances = model_rf.feature_importances_
feat_imp = pd.Series(importances, index=X.columns)

plt.figure(figsize=(10, 6))
feat_imp.sort_values().plot(kind="barh")
plt.title("Random Forest Feature Importance")
plt.tight_layout()
plt.savefig("outputs/feature_importance.png", dpi=150)
plt.show()

model_accuracies = {
    "Logistic Regression": accuracy_score(y_test, y_pred_lr),
    "Decision Tree": accuracy_score(y_test, y_pred_dt),
    "Random Forest": accuracy_score(y_test, y_pred_rf),
    "KNN": accuracy_score(y_test, y_pred_knn),
    "Naive Bayes": accuracy_score(y_test, y_pred_nb),
}

plt.figure(figsize=(10, 6))
plt.bar(model_accuracies.keys(), model_accuracies.values())
plt.xticks(rotation=45)
plt.ylabel("Accuracy")
plt.title("Model Accuracy Comparison")
plt.tight_layout()
plt.savefig("outputs/model_accuracy_comparison.png", dpi=150)
plt.show()

# ---------------------------------------------------------------------------
# 5. Confusion matrices
# ---------------------------------------------------------------------------
for name, y_pred in [
    ("Random Forest", y_pred_rf),
    ("Logistic Regression", y_pred_lr),
    ("Decision Tree", y_pred_dt),
]:
    cm = confusion_matrix(y_test, y_pred)
    print(f"\n{name} confusion matrix:")
    print(cm)

    ConfusionMatrixDisplay(confusion_matrix=cm).plot()
    plt.title(f"{name} Confusion Matrix")
    plt.savefig(f"outputs/confusion_{name.lower().replace(' ', '_')}.png", dpi=150)
    plt.show()

# ---------------------------------------------------------------------------
# 6. Exploratory data analysis
# ---------------------------------------------------------------------------
sns.countplot(x="burnout_level", data=df)
plt.title("Distribution of Burnout Level")
plt.show()

plt.figure(figsize=(12, 8))
sns.heatmap(df.corr(), annot=True, cmap="coolwarm", fmt=".2f")
plt.title("Correlation Heatmap")
plt.tight_layout()
plt.savefig("outputs/correlation_heatmap.png", dpi=150)
plt.show()

df["burnout_score"].hist(bins=20)
plt.title("Distribution of Burnout Score")
plt.xlabel("Burnout Score")
plt.ylabel("Frequency")
plt.show()

sns.boxplot(x="burnout_level", y="burnout_score", data=df)
plt.title("Burnout Score by Burnout Level")
plt.show()

sns.pairplot(df[["burnout_score", "age", "work_hours_per_week", "stress_level"]])
plt.savefig("outputs/pairplot.png", dpi=150)
plt.show()

# ---------------------------------------------------------------------------
# 7. PCA visualization (diagnostic, not used for modeling)
# ---------------------------------------------------------------------------
pca = PCA(n_components=2)
X_pca = pca.fit_transform(X_train_scaled)

plt.figure(figsize=(8, 6))
plt.scatter(X_pca[:, 0], X_pca[:, 1], c=y_train, alpha=0.7)
plt.xlabel("Principal Component 1")
plt.ylabel("Principal Component 2")
plt.title("PCA Visualization of Training Data")
plt.colorbar()
plt.savefig("outputs/pca_visualization.png", dpi=150)
plt.show()

# ---------------------------------------------------------------------------
# 8. K-Means clustering (unsupervised employee profiles)
# ---------------------------------------------------------------------------
kmeans = KMeans(n_clusters=3, random_state=RANDOM_STATE, n_init=10)
df["cluster"] = kmeans.fit_predict(scaler.transform(X))

X_pca_all = PCA(n_components=2).fit_transform(scaler.transform(X))

plt.figure(figsize=(8, 6))
plt.scatter(X_pca_all[:, 0], X_pca_all[:, 1], c=df["cluster"], alpha=0.7)
plt.title("K-Means Clusters Visualization")
plt.xlabel("PCA 1")
plt.ylabel("PCA 2")
plt.savefig("outputs/kmeans_clusters.png", dpi=150)
plt.show()

print("\nCluster sizes:")
print(df["cluster"].value_counts().sort_index())
