"""Train five classifiers on UCI Spambase and save models, test data and metrics."""
import os
import joblib
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.svm import SVC
from metrics_utils import compute_metrics

SEED = 42
os.makedirs("model", exist_ok=True)
names = [l.split(":")[0] for l in open("spambase.names").read().splitlines()
         if l and l[0] not in "|1" and ":" in l and "continuous" in l]
cols = names + ["is_spam"]
df = pd.read_csv("spambase.data", header=None, names=cols)

train, test = train_test_split(df, test_size=0.2, stratify=df["is_spam"], random_state=SEED)
test.to_csv("test_data.csv", index=False)
X_tr, y_tr = train.drop(columns="is_spam"), train["is_spam"]
X_te, y_te = test.drop(columns="is_spam"), test["is_spam"]

models = {
    "Logistic Regression": make_pipeline(StandardScaler(), LogisticRegression(max_iter=2000)),
    "Decision Tree": DecisionTreeClassifier(max_depth=8, random_state=SEED),
    "KNN": make_pipeline(StandardScaler(), KNeighborsClassifier(n_neighbors=7)),
    "Naive Bayes": GaussianNB(),
    "Support Vector Machine": make_pipeline(StandardScaler(), SVC(probability=True, random_state=SEED)),
}
rows = []
for name, m in models.items():
    m.fit(X_tr, y_tr)
    joblib.dump(m, f"model/{name.lower().replace(' ', '_')}.joblib")
    rows.append({"ML Model Name": name, **compute_metrics(m, X_te, y_te)})
res = pd.DataFrame(rows).round(4)
res.to_csv("metrics.csv", index=False)
print(len(df), "rows,", X_tr.shape[1], "features")
print(res.to_string(index=False))
