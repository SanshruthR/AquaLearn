import streamlit as st
import h2o
from h2o.automl import H2OAutoML
import pandas as pd
import os
import numpy as np
from sklearn.metrics import mean_squared_error
import matplotlib.pyplot as plt
import shutil
import zipfile
import io
import tempfile
import zipfile

# Set page config at the very beginning
st.set_page_config(page_title="AquaLearn", layout="wide")

# Initialize the H2O server
h2o.init()
def rename_columns_alphabetically(df):
    new_columns = [chr(65 + i) for i in range(len(df.columns))]
    return df.rename(columns=dict(zip(df.columns, new_columns)))

def sanitize_column_name(name):
    # Replace non-alphanumeric characters with underscores
    sanitized = ''.join(c if c.isalnum() else '_' for c in name)
    # Ensure the name starts with a letter or underscore
    if not sanitized[0].isalpha() and sanitized[0] != '_':
        sanitized = 'f_' + sanitized
    return sanitized

# Create a directory for saving models
if not os.path.exists("saved_models"):
    os.makedirs("saved_models")

def load_data():
    st.title("Aqua Learn")
    uploaded_file = st.file_uploader("Choose a CSV file", type="csv")
    if uploaded_file is not None:
        train = pd.read_csv(uploaded_file)
        st.write(train.head())
        return h2o.H2OFrame(train)
    return None

def select_problem_type():
    return st.selectbox("Select Problem Type:", ['Classification', 'Regression'])

def select_target_column(train_h2o):
    return st.selectbox("Select Target Column:", train_h2o.columns)

def prepare_features(train_h2o, y, problem_type):
    x = train_h2o.columns
    x.remove(y)
    if problem_type == 'Classification':
        train_h2o[y] = train_h2o[y].asfactor()
    
    # Rename columns
    new_columns = [chr(65 + i) for i in range(len(train_h2o.columns))]
    train_h2o.columns = new_columns
    y = new_columns[-1]  # Assume the target is the last column
    x = new_columns[:-1]
    
    return x, y, train_h2o

def select_algorithms():
    algorithm_options = ['DeepLearning', 'GLM', 'GBM', 'DRF', 'XGBoost']
    return st.multiselect("Select Algorithms:", algorithm_options)

def set_automl_parameters():
    max_models = st.number_input("Max Models:", value=20, min_value=1)
    max_runtime = st.number_input("Max Runtime (seconds):", value=600, min_value=1)
    return max_models, max_runtime

def run_automl(x, y, train, problem_type, selected_algos, max_models, max_runtime):
    aml = H2OAutoML(max_models=max_models,
                    seed=1,
                    max_runtime_secs=max_runtime,
                    sort_metric="AUC" if problem_type == 'Classification' else "RMSE",
                    include_algos=selected_algos)
    aml.train(x=x, y=y, training_frame=train)
    return aml

def display_results(aml, test):
    st.subheader("AutoML Leaderboard")
    st.write(aml.leaderboard.as_data_frame())

    st.subheader("Best Model Performance")
    best_model = aml.leader
    perf = best_model.model_performance(test)
    st.write(perf)

def save_and_evaluate_models(aml, test, y, problem_type):
    if st.button("Save Models and Calculate Performance"):
        model_performances = []
        for model_id in aml.leaderboard['model_id'].as_data_frame().values:
            model = h2o.get_model(model_id[0])

            # model_path = os.path.join("saved_models", f"{model_id[0]}")
            # h2o.save_model(model=model, path=model_path, force=True)
            # st.session_state.saved_models.append((model_id[0], model_path))

            preds = model.predict(test)
            actual = test[y].as_data_frame().values.flatten()
            predicted = preds.as_data_frame()['predict'].values.flatten()

            if problem_type == 'Classification':
                performance = (actual == predicted).mean()
                metric_name = 'accuracy'
            else:
                performance = np.sqrt(mean_squared_error(actual, predicted))
                metric_name = 'rmse'

            model_performances.append({'model_id': model_id[0], metric_name: performance})

        performance_df = pd.DataFrame(model_performances)
        st.write(performance_df)

        # Create and display the bar plot
        st.subheader("Model Performance Visualization")
        fig, ax = plt.subplots(figsize=(10, 6))
        performance_df.sort_values(by=metric_name, ascending=False, inplace=True)
        ax.barh(performance_df['model_id'], performance_df[metric_name], color='skyblue')
        ax.set_xlabel(metric_name.capitalize())
        ax.set_ylabel('Model ID')
        ax.set_title(f'Model {metric_name.capitalize()} from H2O AutoML')
        ax.grid(axis='x')
        st.pyplot(fig)

def download_model():
    st.subheader("Download Model")
    if 'saved_models' in st.session_state and st.session_state.saved_models:
        model_to_download = st.selectbox("Select Model to Download:",
                                         [model[0] for model in st.session_state.saved_models])
        if st.button("Download Selected Model"):
            model_path = next(model[1] for model in st.session_state.saved_models if model[0] == model_to_download)

            if os.path.isdir(model_path):
                # If it's a directory, create a zip file
                zip_buffer = io.BytesIO()
                with zipfile.ZipFile(zip_buffer, 'w', zipfile.ZIP_DEFLATED) as zip_file:
                    for root, _, files in os.walk(model_path):
                        for file in files:
                            zip_file.write(os.path.join(root, file),
                                           os.path.relpath(os.path.join(root, file), model_path))

                zip_buffer.seek(0)
                st.download_button(
                    label="Click to Download",
                    data=zip_buffer,
                    file_name=f"{model_to_download}.zip",
                    mime="application/zip"
                )
            else:
                # If it's already a file, offer it for download
                with open(model_path, "rb") as file:
                    st.download_button(
                        label="Click to Download",
                        data=file,
                        file_name=f"{model_to_download}.zip",
                        mime="application/zip"
                    )
    else:
        st.write("No models available for download. Please train and save models first.")

def further_training(aml, x, y, train, problem_type):
    st.subheader("Further Training")
    leaderboard_df = aml.leaderboard.as_data_frame()
    model_to_train = st.selectbox("Select Model for Training:", leaderboard_df['model_id'].tolist())
    training_time = st.number_input("Training Time (seconds):", value=60, min_value=1)

    if st.button("Train Model"):
        model = h2o.get_model(model_to_train)

        with st.spinner(f"Training model: {model_to_train} for {training_time} seconds..."):
            if isinstance(model, h2o.estimators.stackedensemble.H2OStackedEnsembleEstimator):
                aml = H2OAutoML(max_runtime_secs=training_time, seed=1, sort_metric="AUC" if problem_type == 'Classification' else "RMSE")
                aml.train(x=x, y=y, training_frame=train)
                model = aml.leader
            else:
                model.train(x=x, y=y, training_frame=train, max_runtime_secs=training_time)

        perf = model.model_performance(train)
        st.write("Model performance after training:")
        st.write(perf)

        # Create a temporary directory to save the model
        temp_dir = os.path.join("saved_models", "temp")
        os.makedirs(temp_dir, exist_ok=True)
        model_path = os.path.join(temp_dir, f"{model.model_id}")
        h2o.save_model(model=model, path=model_path, force=True)

        # Create a zip file of the model
        zip_buffer = io.BytesIO()
        with zipfile.ZipFile(zip_buffer, 'w', zipfile.ZIP_DEFLATED) as zip_file:
            for root, _, files in os.walk(model_path):
                for file in files:
                    zip_file.write(os.path.join(root, file),
                                   os.path.relpath(os.path.join(root, file), model_path))

        zip_buffer.seek(0)
        st.download_button(
            label="Download Retrained Model",
            data=zip_buffer,
            file_name=f"{model.model_id}.zip",
            mime="application/zip"
        )

        # Clean up the temporary directory
        shutil.rmtree(temp_dir)

        st.success(f"Retrained model ready for download: {model.model_id}")

def make_prediction():
    st.subheader("Make Prediction")

    uploaded_zip = st.file_uploader("Upload a zip file containing the model", type="zip")
    if uploaded_zip is not None:
        with tempfile.TemporaryDirectory() as tmpdirname:
            zip_path = os.path.join(tmpdirname, "model.zip")
            with open(zip_path, "wb") as f:
                f.write(uploaded_zip.getbuffer())

            with zipfile.ZipFile(zip_path, 'r') as zip_ref:
                zip_ref.extractall(tmpdirname)

            extracted_files = os.listdir(tmpdirname)
            if len(extracted_files) == 0:
                st.error("The uploaded zip file is empty.")
                return

            model_file = next((f for f in extracted_files if f != "model.zip"), None)
            if model_file is None:
                st.error("No model file found in the uploaded zip.")
                return

            model_path = os.path.join(tmpdirname, model_file)

            try:
                model_for_prediction = h2o.load_model(model_path)
            except Exception as e:
                st.error(f"Error loading the model: {str(e)}")
                st.error("Please ensure you're uploading a valid H2O model file.")
                return

            # Ask user to input feature names
            feature_names_input = st.text_input("Enter feature names, separated by commas:")
            original_feature_names = [name.strip() for name in feature_names_input.split(',') if name.strip()]
            
            if not original_feature_names:
                st.error("Please enter at least one feature name.")
                return

            # Create a mapping from original names to A, B, C, etc.
            feature_mapping = {name: chr(65 + i) for i, name in enumerate(original_feature_names)}
            reverse_mapping = {v: k for k, v in feature_mapping.items()}

            prediction_type = st.radio("Choose prediction type:", ["Upload CSV", "Single Entry"])

            if prediction_type == "Upload CSV":
                uploaded_csv = st.file_uploader("Upload a CSV file for prediction", type="csv")
                if uploaded_csv is not None:
                    prediction_data = pd.read_csv(uploaded_csv)
                    
                    # Rename columns to A, B, C, etc.
                    prediction_data = prediction_data.rename(columns=feature_mapping)
                    
                    prediction_h2o = h2o.H2OFrame(prediction_data)
                    try:
                        predictions = model_for_prediction.predict(prediction_h2o)
                        predictions_df = predictions.as_data_frame()

                        # Combine original data with predictions
                        result_df = pd.concat([prediction_data, predictions_df], axis=1)

                        # Rename columns back to original names for display
                        result_df = result_df.rename(columns=reverse_mapping)

                        st.write("Predictions (showing first 10 rows):")
                        st.write(result_df.head(10))

                        # Option to download the full results
                        csv = result_df.to_csv(index=False)
                        st.download_button(
                            label="Download full results as CSV",
                            data=csv,
                            file_name="predictions_results.csv",
                            mime="text/csv"
                        )
                    except Exception as e:
                        st.error(f"Error making predictions: {str(e)}")
                        st.error("Please ensure your CSV file matches the model's expected input format.")

            else:  # Single Entry
                sample_input = {}
                for original_name, coded_name in feature_mapping.items():
                    value = st.text_input(f"Enter {original_name} ({coded_name}):")
                    try:
                        sample_input[coded_name] = [float(value)]
                    except ValueError:
                        sample_input[coded_name] = [value]

                if st.button("Predict"):
                    sample_h2o = h2o.H2OFrame(sample_input)
                    try:
                        predictions = model_for_prediction.predict(sample_h2o)
                        prediction_value = predictions['predict'][0,0]
                        st.write(f"Predicted value: {prediction_value}")
                    except Exception as e:
                        st.error(f"Error making prediction: {str(e)}")
                        st.error("Please ensure you've entered valid input values.")
    else:
        st.write("Please upload a zip file containing the model to make predictions.")
def main():
    train_h2o = load_data()
    if train_h2o is not None:
        problem_type = select_problem_type()
        target_column = select_target_column(train_h2o)

        if st.button("Set Target and Continue"):
          x, target_column, train_h2o = prepare_features(train_h2o, target_column, problem_type)
          st.session_state.features_prepared = True
          st.session_state.x = x
          st.session_state.target_column = target_column
          st.session_state.train_h2o = train_h2o
          st.session_state.problem_type = problem_type

    if 'features_prepared' in st.session_state and st.session_state.features_prepared:
        st.write(f"Target Column: {st.session_state.target_column}")
        st.write(f"Feature Columns: {st.session_state.x}")

        train, test = st.session_state.train_h2o.split_frame(ratios=[0.8])

        selected_algos = select_algorithms()
        max_models, max_runtime = set_automl_parameters()

        if st.button("Start AutoML"):
            if not selected_algos:
                st.error("Please select at least one algorithm.")
            else:
                with st.spinner("Running AutoML..."):
                    aml = run_automl(st.session_state.x, st.session_state.target_column, train,
                                     st.session_state.problem_type, selected_algos, max_models, max_runtime)

                st.success("AutoML training completed.")
                st.session_state.aml = aml
                st.session_state.test = test

        if 'aml' in st.session_state:
            display_results(st.session_state.aml, st.session_state.test)
            save_and_evaluate_models(st.session_state.aml, st.session_state.test, st.session_state.target_column, st.session_state.problem_type)
            download_model()
            further_training(st.session_state.aml, st.session_state.x, st.session_state.target_column, train, st.session_state.problem_type)

    make_prediction()  # Call make_prediction without arguments

if __name__ == "__main__":
    if 'features_prepared' not in st.session_state:
        st.session_state.features_prepared = False
    if 'saved_models' not in st.session_state:
        st.session_state.saved_models = []
    main()

# Clean up saved models when the script ends
shutil.rmtree("saved_models", ignore_errors=True)
