import pandas as pd
import numpy as np

df = pd.read_csv("rainfall_newdf.csv")

from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

# Models
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier, AdaBoostClassifier
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from xgboost import XGBClassifier
from lightgbm import LGBMClassifier
from catboost import CatBoostClassifier

X = df.drop(columns=["rainfall"])  
y = df["rainfall"]

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

models = {
    "Logistic Regression": LogisticRegression(),
    "Decision Tree": DecisionTreeClassifier(),
    "Random Forest": RandomForestClassifier(),
    "Gradient Boosting": GradientBoostingClassifier(),
    "AdaBoost": AdaBoostClassifier(),
    "SVM": SVC(probability=True),
    "k-NN": KNeighborsClassifier(),
    "Naive Bayes": GaussianNB(),
    "XGBoost": XGBClassifier(use_label_encoder=False, eval_metric='logloss'),
    "LightGBM": LGBMClassifier(),
    "CatBoost": CatBoostClassifier(verbose=0)
}

results = []

for name, model in models.items():
    pipeline = Pipeline([  
        ("scaler", StandardScaler()),
        ("classifier", model)
    ])

    pipeline.fit(X_train, y_train)

    y_pred = pipeline.predict(X_test)

    accuracy = accuracy_score(y_test, y_pred)
    precision = precision_score(y_test, y_pred)
    recall = recall_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred)

    results.append({"Model": name, "Accuracy": accuracy, "Precision": precision, "Recall": recall, "F1 Score": f1})

results_df = pd.DataFrame(results).sort_values(by="Accuracy", ascending=False)


# rf_model = RandomForestClassifier(random_state=42)

pipe = Pipeline([
    ("scaler", StandardScaler()),  
    ("clf", RandomForestClassifier(random_state=42))  # Replace with your model
])

param_grid_rf = {
    "clf__n_estimators": [50, 200, 250, 300],  # Prefix 'clf__' to refer to model inside Pipeline
    "clf__max_features": ['sqrt', 'log2'],
    "clf__max_depth": [None, 2, 4, 7],
    "clf__min_samples_split": [1, 2, 6],  
    "clf__min_samples_leaf": [1, 2, 3, 4]
}

from sklearn.model_selection import GridSearchCV

grid_search_rf = GridSearchCV(estimator=pipe, param_grid=param_grid_rf, cv = 5, n_jobs = -1, verbose = 2)

grid_search_rf.fit(X_train, y_train)

best_rf_model = grid_search_rf.best_estimator_

from sklearn.metrics import classification_report

report_dict = classification_report(y_test, y_pred, output_dict = True)

from sklearn.metrics import ConfusionMatrixDisplay

cm = ConfusionMatrixDisplay.from_predictions(y_test,y_pred)

import mlflow
import dagshub
dagshub.init(repo_owner='ajstyle007', repo_name='my-first-repo', mlflow=True)


mlflow.set_experiment("Rainfall")
mlflow.set_tracking_uri(uri="https://dagshub.com/ajstyle007/my-first-repo.mlflow")

with mlflow.start_run(run_name="rainfall6"):
    mlflow.log_params(grid_search_rf.best_params_)
    mlflow.log_metrics({
        "Accuracy" : report_dict["accuracy"],
        "Precision_0" : report_dict["0"]["precision"],
        "Recall_0" : report_dict["0"]["recall"],

        "Precision_1" : report_dict["1"]["precision"],
        "Recall_1" : report_dict["1"]["recall"],
        "f1_score_macro":report_dict['macro avg']['f1-score']
        
    })
    # log confusion matrix
    mlflow.log_figure(figure=cm.figure_, artifact_file="confusion_matrix.png")
    mlflow.sklearn.log_model(best_rf_model, "Random Forest Model", registered_model_name="RainFall_RF_model")

model_name = "RainFall_RF_model"
current_model_uri = f"models:/{model_name}@champion"
production_model_name = 'rainfall-prediction-production'

client = mlflow.MlflowClient()
client.copy_model_version(src_model_uri=current_model_uri, dst_name=production_model_name)