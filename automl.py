import streamlit as st
import pandas as pd
import os
import sqlite3
from ydata_profiling import ProfileReport
from streamlit_pandas_profiling import st_profile_report
from sklearn.model_selection import train_test_split
from models import evaluate_scikit_models, evaluate_pycaret

# Directories
PROFILES_DIR = "profiles"
MODELS_DIR = "models"

os.makedirs(PROFILES_DIR, exist_ok=True)
os.makedirs(MODELS_DIR, exist_ok=True)

# Database Connection
def init_db(filename):
    conn = sqlite3.connect(f"{filename}.db")
    return conn

# Sidebar Navigation
st.sidebar.title("Navigation")
pages = ["Upload CSV", "Data Analysis", "Machine Learning", "Predict"]
st.session_state["page"] = st.sidebar.radio("Go to", pages, index=0)

if "df" not in st.session_state:
    st.session_state["df"] = None
if "selected_features" not in st.session_state:
    st.session_state["selected_features"] = []
if "target" not in st.session_state:
    st.session_state["target"] = None

# Page 1: Upload CSV
if st.session_state["page"] == "Upload CSV":
    st.title("Upload CSV File")
    uploaded_file = st.file_uploader("Choose a CSV file", type="csv")

    if uploaded_file is not None:
        df = pd.read_csv(uploaded_file)
        st.session_state["df"] = df
        filename = uploaded_file.name.split(".")[0]
        st.session_state["uploaded_filename"] = filename
        conn = init_db(filename)
        df.to_sql("data", conn, if_exists="replace", index=False)
        st.success(f"File uploaded and stored as {filename}.db")
        
    if st.session_state["df"] is not None:
        st.dataframe(st.session_state["df"])
    else:
        st.warning("Please upload a CSV file first.")

# Page 2: Data Analysis
elif st.session_state["page"] == "Data Analysis":
    st.title("Data Analysis")
    if "df" in st.session_state:
        df = st.session_state["df"]
        profile = ProfileReport(df, title="Pandas Profiling Report", explorative=True)
        st_profile_report(profile)

        filename = st.text_input("Enter filename to save profile report", "profile")
        if st.button("Save Profile Report"):
            profile_path = os.path.join(PROFILES_DIR, f"{filename}.html")
            profile.to_file(profile_path)
            st.success(f"Profile saved to {profile_path}")
    else:
        st.warning("Please upload a CSV file first.")
   

# Page 3: Machine Learning
elif st.session_state["page"] == "Machine Learning":
    st.title("Machine Learning")

    if st.session_state["df"] is not None:
        df = st.session_state["df"]
        target = st.selectbox("Select Target Column", df.columns)
        features = st.multiselect("Select Features", [col for col in df.columns if col != target])

        if len(features) > 0:
            st.session_state["selected_features"] = features
            st.session_state["target"] = target

        ml_option = st.radio("Select Framework", ["Scikit-Learn", "PyCaret"])

        if st.button("Run Models"):
            X = df[features]
            y = df[target]
            X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

            if ml_option == "Scikit-Learn":
                results = evaluate_scikit_models(X_train, X_test, y_train, y_test)
                results_df = pd.DataFrame(results)
                st.dataframe(results_df)

                # Save best model
                best_result = max(results, key=lambda x: x["Accuracy"])
                best_model = best_result["Trained Model"]
                st.session_state["best_model"] = best_model
                st.success(f"Best Model Trained: {best_result['Model']}")

            elif ml_option == "PyCaret":
                pycaret_results, best_models = evaluate_pycaret(df[features + [target]], target)
                st.dataframe(pycaret_results)

                # Save best PyCaret model
                st.session_state["best_pycaret_model"] = best_models[0]
                st.success("Best PyCaret Model Trained.")
    else:
        st.warning("Please upload a CSV file first.")

# Page 4: Predict
elif st.session_state["page"] == "Predict":
    st.title("Predict")

    if (
        "best_model" in st.session_state or "best_pycaret_model" in st.session_state
    ) and st.session_state["selected_features"]:
        st.write("Provide values for the following features:")
        input_data = {
            feature: st.number_input(f"Enter value for {feature}", value=0.0)
            for feature in st.session_state["selected_features"]
        }
        input_df = pd.DataFrame([input_data])

        if st.button("Predict"):
            if "best_model" in st.session_state:
                model = st.session_state["best_model"]
                prediction = model.predict(input_df)
                st.success(f"Predicted Value: {prediction[0]}")
            elif "best_pycaret_model" in st.session_state:
                from pycaret.classification import predict_model

                model = st.session_state["best_pycaret_model"]
                prediction = predict_model(model, data=input_df)
                st.success(f"Predicted Value: {prediction['Label'].iloc[0]}")
    else:
        st.warning(
            "No trained model found or features not selected. Please train a model first."
        )
