import numpy as np
import pandas as pd
from sklearn import metrics
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import GridSearchCV, train_test_split
from sklearn.ensemble import BaggingClassifier, GradientBoostingClassifier

class Logistic:
    def __init__(self,
                c_min=0.01, c_max=100, c_interval=10,
                l1_min=0.1, l1_max=1.0, l1_interval=0.1,
                alpha_min=0.1, alpha_max=0.9, alpha_interval=0.1,
                penalty='l2'  # 'l1', 'l2', or 'elasticnet'
                ):
        # Set-Up Parameters based on User Input
        c_values = np.linspace(c_min, c_max + c_interval, int((c_max - c_min) / c_interval) + 1)
        l1_ratios = np.linspace(l1_min, l1_max + l1_interval, int((l1_max - l1_min) / l1_interval) + 1)
        alpha_values = np.linspace(alpha_min, alpha_max + alpha_interval, int((alpha_max - alpha_min) / alpha_interval) + 1)
        n_estimators = [50, 100]

        self.penalty = penalty.lower()
        self.parameters = {}
        base_estimator = None

        if self.penalty == 'l1':
            base_estimator = LogisticRegression(penalty='l1', solver='saga', max_iter=10000)
            self.parameters = {
                'estimator__C': c_values,
                'n_estimators': n_estimators
            }
        elif self.penalty == 'l2':
            base_estimator = LogisticRegression(penalty='l2', solver='saga', max_iter=10000)
            self.parameters = {
                'estimator__C': c_values,
                'n_estimators': n_estimators
            }
        elif self.penalty == 'elasticnet':
            base_estimator = LogisticRegression(penalty='elasticnet', solver='saga', max_iter=10000)
            self.parameters = {
                'estimator__C': c_values,
                'estimator__l1_ratio': l1_ratios,
                'n_estimators': n_estimators
            }
        else:
            raise ValueError("penalty must be one of 'l1', 'l2', or 'elasticnet'")

        # Initialize GridSearchCV
        self.grid = GridSearchCV(
            estimator=BaggingClassifier(base_estimator),
            param_grid=self.parameters,
            cv=5,
            verbose=1,
            n_jobs=-1
        )
        
    def split_train_test(self, df):
        """
        Splits the input DataFrame into train and test sets.
        Expects columns: 'Pregnancies', 'Glucose', 'Blood Pressure', 'Skin Thickness', 'Insulin',
        'Body Mass Index', 'Diabetes Pedigree Function', 'Age', 'Outcome'.
        Returns: X_train, X_test, y_train, y_test
        """
        feature_cols = [
            'Pregnancies', 'Glucose', 'Blood Pressure', 'Skin Thickness',
            'Insulin', 'Body Mass Index', 'Diabetes Pedigree Function', 'Age'
        ]
        X = df[feature_cols]
        y = df['Outcome']
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.3, random_state=42, stratify=y
        )
        return X_train, X_test, y_train, y_test

    def train_and_evaluate(self, X_train, X_test, y_train, y_test):
        """
        Trains the model using GridSearchCV, evaluates on train and test sets, and returns results as a DataFrame.
        """
        # Fit model
        self.grid.fit(X_train, y_train)

        # Predict
        y_train_pred = self.grid.predict(X_train)
        y_test_pred = self.grid.predict(X_test)
        y_train_prob = self.grid.predict_proba(X_train)[:, 1]
        y_test_prob = self.grid.predict_proba(X_test)[:, 1]

        # Metrics
        results = {}
        for split, y_true, y_pred, y_prob in [
            ("Train", y_train, y_train_pred, y_train_prob),
            ("Test", y_test, y_test_pred, y_test_prob)
        ]:
            results[split] = {
                "Accuracy": metrics.accuracy_score(y_true, y_pred),
                "Precision": metrics.precision_score(y_true, y_pred, zero_division=0),
                "Recall": metrics.recall_score(y_true, y_pred, zero_division=0),
                "F1": metrics.f1_score(y_true, y_pred, zero_division=0),
                "ROC AUC": metrics.roc_auc_score(y_true, y_prob),
            }
        df_results = pd.DataFrame(results).T
        return df_results

