# Email Spam Classification - ML Assignment 2

## a. Problem statement
Build and compare five classification models that decide whether an e-mail is **spam (1)** or **not spam (0)** from word/character frequency features, evaluate them with six metrics, and serve the comparison through a Streamlit web app.

## b. Dataset description
- **Source:** UCI Machine Learning Repository - Spambase (https://archive.ics.uci.edu/dataset/94/spambase)
- **Instances:** 4601 (1813 spam, 2788 non-spam) | **Features:** 57 numeric | **Target:** `is_spam` (binary)
- Features: 48 `word_freq_*` (% of words matching a term), 6 `char_freq_*` (% of characters matching `; ( [ ! $ #`), and 3 capital-letter run-length statistics.
- **Split:** 80/20 stratified (random_state=42). 3680 rows for training; 921 rows saved as `test_data.csv` for evaluation and app upload.
- **Preprocessing:** `StandardScaler` inside the pipelines for Logistic Regression, KNN and SVM (saved with the model). Decision Tree and Naive Bayes use raw features.

## c. GitHub repository link
https://github.com/AJAYKUMAR-247/ml-assignment-2

Live app: STREAMLIT_LINK_HERE

Repo layout: `app.py`, `train.py`, `metrics_utils.py`, `requirements.txt`, `README.md`, `test_data.csv`, `spambase.data`/`spambase.names` (raw data), `metrics.csv`, `model/*.joblib`.

## d. Models used
Metrics on the held-out test set (921 rows). Precision, Recall and F1 are for the spam class.

| ML Model Name | Accuracy | AUC | Precision | Recall | F1 | MCC |
|---|---|---|---|---|---|---|
| Logistic Regression | 0.9294 | 0.9702 | 0.9209 | 0.8981 | 0.9093 | 0.8518 |
| Decision Tree | 0.9034 | 0.9154 | 0.9029 | 0.8457 | 0.8734 | 0.7965 |
| KNN | 0.9088 | 0.9566 | 0.8930 | 0.8733 | 0.8830 | 0.8084 |
| Naive Bayes | 0.8339 | 0.9449 | 0.7178 | 0.9532 | 0.8189 | 0.6941 |
| Support Vector Machine | 0.9273 | 0.9667 | 0.9277 | 0.8843 | 0.9055 | 0.8471 |

### Observations

| ML Model Name | Observation about model performance |
|---|---|
| Logistic Regression | Best overall: highest accuracy, AUC, F1 and MCC. The classes are close to linearly separable in the scaled feature space, and the model is fast and well calibrated. |
| Decision Tree | Weakest of the discriminative models (AUC 0.915). Depth was capped at 8 to limit overfitting, but a single tree still misses more spam (recall 0.846) and ranks emails less well than the smoother models. |
| KNN | Solid (accuracy 0.909, AUC 0.957) once features are scaled, but below LR and SVM. It is sensitive to the 57-dimensional space and is the slowest at prediction time. |
| Naive Bayes | Lowest accuracy and precision (0.718), but the highest recall (0.953). The independence assumption is violated by correlated word frequencies, so it over-flags spam. It suits cases where missing spam costs more than false alarms. |
| Support Vector Machine | Very close to Logistic Regression, with the best precision (0.928) and a high AUC of 0.967. The RBF kernel adds little over the linear boundary here, and training is slower. |
| **Overall winner** | **Logistic Regression** - best Accuracy, AUC, F1 and MCC. SVM is a close second and has the best precision. |

## How to run
```
pip install -r requirements.txt
python train.py          # retrains and regenerates model/, test_data.csv, metrics.csv
streamlit run app.py
```
The app lets you upload a test CSV (features + `is_spam`), pick a model, and see the metrics, confusion matrix, classification report and a five-model comparison. Models are loaded from `model/` and never retrained.
