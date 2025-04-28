import torch
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.metrics import accuracy_score
from tabpfn_client import init, TabPFNClassifier
import os
os.environ["TABPFN_ALLOW_CPU_LARGE_DATASET"] = "1"

def TabPFN_model(X_train, X_test, y_train, y_test):
    # clf = TabPFNClassifier()
    # clf.fit(X_train, y_train)
    # y_pred = clf.predict(X_test)
    # test_accuracy = clf.score(X_test, y_pred)
    # print(f"Accuracy: {test_accuracy:.2f}")

    # clf = AutoTabPFNClassifier(device="cuda" if torch.cuda.is_available() else "cpu")
    param_grid = {
        'n_estimators': [8, 16, 32],
        'softmax_temperature': [0.7, 1.0, 1.3]
    }
    grid_search = GridSearchCV(TabPFNClassifier(), param_grid, cv=3)

    grid_search.fit(X_train, y_train)

    print(f"Best parameters: {grid_search.best_params_}")
    y_grid_pred = grid_search.predict(X_test)
    grid_accuracy = accuracy_score(y_test, y_grid_pred)
    print(grid_accuracy)

