# Predicting Employee Burnout in the Tech Industry

A multi-model machine learning project that classifies employee burnout levels (low, medium, high) from behavioral, psychological, and organizational survey data. Five classifiers are compared, and the analysis identifies the main drivers of burnout and profiles employees with PCA and K-Means clustering.

Built as a Data Mining project at Tunis Business School (BA360), 2025–2026.

## Research Questions

1. Which factors most strongly predict burnout?
2. Can machine learning models accurately classify burnout level?
3. Which algorithm performs best, and does model complexity pay off?
4. Do burnout profiles form distinct clusters in feature space?
5. How much do support factors (manager, social, work-life balance) moderate burnout risk?
6. Is burnout a demographic or structural problem?

## Dataset

- **Source:** [Tech Mental Health & Burnout Dataset (Kaggle)](https://www.kaggle.com/datasets/suhanigupta04/employee-mental-health-and-burnout-dataset)
- **Size:** ~150,000 records, 25 variables
- **Note:** The dataset is simulated. Results show the approach works on this data, not that it will transfer directly to real organizations.
- **Location:** `data/tech_mental_health_burnout.csv`. If the file is missing, download it from the Kaggle link above and place it in `data/`.

Variables span four domains: demographic (age, gender, job role, experience), organizational (work hours, manager support, work-life balance, job satisfaction), behavioral (sleep, physical activity, screen time, social support), and psychological (stress, anxiety, depression). The target is `burnout_level` (0 = low, 1 = medium, 2 = high).

## Methodology

1. **Preprocessing**
   - Numeric missing values imputed with the median; categorical with the mode
   - Label encoding for categorical variables
   - `burnout_score` removed from features to prevent data leakage (r ≈ 0.71 with the target)
   - 80/20 stratified train-test split, `random_state=42`
   - `StandardScaler` for Logistic Regression, KNN, and Naive Bayes
2. **Exploratory data analysis:** distributions, correlation heatmap, pairplots
3. **Classification models**
   - Logistic Regression (baseline)
   - Decision Tree (`max_depth=5`)
   - Random Forest (100 trees)
   - K-Nearest Neighbors (`k=5`)
   - Gaussian Naive Bayes
4. **Dimensionality reduction:** PCA with 2 components, for visualization and separability checks
5. **Clustering:** K-Means with 3 clusters, for employee profiling

## Results

Test accuracy from the original run (see note below):

| Model | Test Accuracy |
|---|---|
| Logistic Regression | ~93% |
| Random Forest | ~93% |
| Decision Tree | ~91% |
| KNN | ~91% |
| Naive Bayes | ~89% |

**Top burnout predictors (Random Forest feature importance):** stress level, depression score, work hours per week, anxiety score, work-life balance.

**Key findings**
- Psychological indicators carry the most predictive weight.
- Employees working 60+ hours per week are over-represented in the high-burnout group.
- Manager support is the strongest protective factor among support variables.
- Age, gender, and job role show near-zero correlation with burnout.
- Logistic Regression matches Random Forest, suggesting the decision boundary is approximately linear.

**Limitations**
- Accuracy is inflated by the dominant medium class. Per-class precision, recall, and F1 should be read alongside it, especially for the high-burnout class.
- The low-burnout class is very small in the test set.
- The data is cross-sectional (one time point) and simulated.
- Label encoding can imply false ordering for nominal categories. One-hot encoding is preferable in production.

> **Note:** The script was updated to use stratified splitting and to fix a Random Forest report bug. Rerun it to get the exact current numbers, and update the table above and the report if they differ.

## Project Structure

```
├── data/          # Dataset (tech_mental_health_burnout.csv)
├── src/           # Analysis script (burnout_models.py)
├── docs/          # Project report and presentation (PDF)
└── outputs/       # Generated figures (created when the script runs)
```

## How to Run

From the project root:

```bash
pip install -r requirements.txt
python src/burnout_models.py
```

The script prints metrics for each model and saves figures to `outputs/`.

## Tools

Python (pandas, NumPy, scikit-learn, matplotlib, seaborn)

## Team

- Alaa Azouzi
- Sarra Alioua
- Nouha Boukhris
- Mariem Chammem

Supervised by Dr. Aloui Donia, Tunis Business School, 2025–2026.

## License

Code is released under the [MIT License](LICENSE). The dataset is subject to its own Kaggle license.
