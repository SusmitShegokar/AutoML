# import time
# from sklearn.linear_model import LogisticRegression
# from sklearn.tree import DecisionTreeClassifier
# from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier
# from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

def evaluate_scikit_models(X_train, X_test, y_train, y_test):
    from sklearn.linear_model import LogisticRegression
    from sklearn.tree import DecisionTreeClassifier
    from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier
    from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
    import time

    models = {
        "Logistic Regression": LogisticRegression(),
        "Decision Tree": DecisionTreeClassifier(),
        "Random Forest": RandomForestClassifier(),
        "Gradient Boosting": GradientBoostingClassifier(),
    }

    results = []
    for name, model in models.items():
        start_time = time.time()
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)
        end_time = time.time()

        metrics = {
            "Model": name,
            "Accuracy": accuracy_score(y_test, y_pred),
            "Precision": precision_score(y_test, y_pred, average="weighted"),
            "Recall": recall_score(y_test, y_pred, average="weighted"),
            "F1-Score": f1_score(y_test, y_pred, average="weighted"),
            "Runtime (s)": round(end_time - start_time, 2),
            "Trained Model": model,  # Include the trained model
        }
        results.append(metrics)

    return results



def evaluate_pycaret(df, target):
    from pycaret.classification import setup, compare_models, pull
    clf_setup = setup(data=df, target=target, session_id=42)  # Ensure `silent=True` is added
    best_models = compare_models(n_select=5)
    results_df = pull()  # Pull results to display as a dataframe
    return results_df, best_models

