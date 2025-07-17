import streamlit as st
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score, mean_squared_error, accuracy_score
from sklearn.linear_model import LinearRegression, Lasso, Ridge, ElasticNet, LogisticRegression
from sklearn.preprocessing import PolynomialFeatures, StandardScaler, LabelEncoder
from sklearn.tree import DecisionTreeRegressor, DecisionTreeClassifier
from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier, GradientBoostingRegressor, GradientBoostingClassifier, ExtraTreesRegressor, ExtraTreesClassifier
from sklearn.neighbors import KNeighborsRegressor, KNeighborsClassifier
from sklearn.svm import SVR, SVC
from sklearn.naive_bayes import GaussianNB
import xgboost as xgb
import numpy as np

# Define AI agent class
class AgentAI:
    def __init__(self):
        # Initialize StandardScaler for numerical feature scaling
        self.scaler = StandardScaler()
        # Initialize LabelEncoder for target variable encoding in classification
        self.label_encoder = LabelEncoder()
        # Store trained models and their associated preprocessing objects (e.g., PolynomialFeatures)
        self.trained_models = {}

        # Define available models for Regression and Classification tasks
        # Added random_state for reproducibility
        self.models = {
            'Regression': {
                'Linear Regression': LinearRegression(),
                'Polynomial Regression': LinearRegression(), # PolynomialFeatures applied externally
                'Lasso Regression': Lasso(random_state=42),
                'Ridge Regression': Ridge(random_state=42),
                'Elastic Net Regression': ElasticNet(random_state=42),
                'Decision Tree Regression': DecisionTreeRegressor(random_state=42),
                'Random Forest Regression': RandomForestRegressor(random_state=42),
                'Extra Trees Regressor': ExtraTreesRegressor(random_state=42),
                'Gradient Boosting Regressor': GradientBoostingRegressor(random_state=42),
                'XGBoost Regressor': xgb.XGBRegressor(random_state=42),
                'KNN Regressor': KNeighborsRegressor(),
                'SVR': SVR()
            },
            'Classification': {
                'Logistic Regression': LogisticRegression(max_iter=2000, random_state=42), # Increased max_iter for convergence
                'Decision Tree Classifier': DecisionTreeClassifier(random_state=42),
                'Random Forest Classifier': RandomForestClassifier(random_state=42),
                'Extra Trees Classifier': ExtraTreesClassifier(random_state=42),
                'Gradient Boosting Classifier': GradientBoostingClassifier(random_state=42),
                'XGBoost Classifier': xgb.XGBClassifier(use_label_encoder=False, eval_metric='logloss', random_state=42), # Suppress warning, set eval_metric
                'KNN Classifier': KNeighborsClassifier(), # Corrected from KNeighborsRegressor
                'SVC': SVC(max_iter=2000, random_state=42), # Increased max_iter for convergence
                'Gaussian Naive Bayes': GaussianNB() # General purpose Naive Bayes
            }
        }

    def _preprocess_data(self, df, target_column, task, fit_scaler_encoder=True):
        """
        Handles missing values, categorical encoding, and feature scaling.
        If fit_scaler_encoder is True, it fits the scaler and encoder.
        Otherwise, it uses the already fitted scaler and encoder for transformation.
        Returns preprocessed X, y, and the list of feature columns.
        """
        df_processed = df.copy()

        # Separate features (X) and target (y)
        if target_column not in df_processed.columns:
            st.error(f"Target column '{target_column}' not found in the dataset.")
            return None, None, None, None

        y = df_processed[target_column]
        X = df_processed.drop(columns=[target_column])

        # Store original categorical columns for consistent one-hot encoding later
        self.original_categorical_cols = X.select_dtypes(include='object').columns.tolist()

        # Handle missing values
        # For numerical columns, fill with mean (from training data if possible, else current data)
        numerical_cols = X.select_dtypes(include=np.number).columns
        for col in numerical_cols:
            if X[col].isnull().any():
                # In a real scenario, you'd use the mean from the training set.
                # For this app, we'll use the mean of the current X.
                X[col] = X[col].fillna(X[col].mean())

        # For categorical columns, fill with mode (from training data if possible, else current data)
        categorical_cols = X.select_dtypes(include='object').columns
        for col in categorical_cols:
            if X[col].isnull().any():
                # In a real scenario, you'd use the mode from the training set.
                X[col] = X[col].fillna(X[col].mode()[0])

        # Categorical encoding for features (One-Hot Encoding)
        # Convert all categorical columns to 'category' dtype first to handle unseen categories gracefully
        for col in self.original_categorical_cols:
            if col in X.columns:
                X[col] = X[col].astype('category')

        X = pd.get_dummies(X, columns=self.original_categorical_cols, drop_first=True)

        # Feature Scaling for numerical features in X
        numerical_cols_after_dummies = X.select_dtypes(include=np.number).columns
        if len(numerical_cols_after_dummies) > 0:
            if fit_scaler_encoder:
                X[numerical_cols_after_dummies] = self.scaler.fit_transform(X[numerical_cols_after_dummies])
            else:
                X[numerical_cols_after_dummies] = self.scaler.transform(X[numerical_cols_after_dummies])

        # Encode target variable if classification and not already numerical
        if task == 'Classification' and y.dtype == 'object':
            if fit_scaler_encoder:
                y = self.label_encoder.fit_transform(y)
                st.info(f"Target variable encoded. Original classes: {self.label_encoder.classes_}")
            else:
                # For prediction, we need to ensure the target variable is known to the encoder,
                # but we don't fit it again. This part is mostly for consistency, as y is not
                # directly used in prediction input.
                pass # y is the output, not input for prediction

        return X, y, X.columns.tolist(), self.original_categorical_cols

    def train_model(self, X_train, y_train, task, model_name, degree=2):
        """
        Trains a specified model and stores it.
        """
        model = self.models[task][model_name]
        try:
            if task == 'Regression' and model_name == 'Polynomial Regression':
                poly_features = PolynomialFeatures(degree=degree)
                X_train_poly = poly_features.fit_transform(X_train)
                model.fit(X_train_poly, y_train)
                self.trained_models[model_name] = {'model': model, 'poly_features': poly_features}
            else:
                model.fit(X_train, y_train)
                self.trained_models[model_name] = {'model': model}
            return True
        except Exception as e:
            st.error(f"Error training {model_name}: {e}")
            return False

    def predict(self, X_test, task, model_name):
        """
        Makes predictions using a specified model.
        """
        if model_name not in self.trained_models:
            st.error(f"Model '{model_name}' has not been trained yet.")
            return None

        model_info = self.trained_models[model_name]
        model = model_info['model']

        try:
            if task == 'Regression' and model_name == 'Polynomial Regression':
                poly_features = model_info['poly_features']
                X_test_poly = poly_features.transform(X_test)
                return model.predict(X_test_poly)
            else:
                return model.predict(X_test)
        except Exception as e:
            st.error(f"Error predicting with {model_name}: {e}")
            return None

    def evaluate(self, y_test, y_pred, task):
        """
        Evaluates model performance based on the task.
        """
        if y_pred is None:
            return -np.inf # Return negative infinity if prediction failed
        try:
            if task == 'Regression':
                return r2_score(y_test, y_pred)
            else:
                return accuracy_score(y_test, y_pred)
        except Exception as e:
            st.error(f"Error evaluating model: {e}")
            return -np.inf

# --- Caching the AgentAI instance ---
@st.cache_resource
def get_agent_ai():
    """Caches and returns the AgentAI instance."""
    return AgentAI()

agent = get_agent_ai()

# --- Caching the ML Pipeline results ---
@st.cache_data(show_spinner="Running AI Agent Analysis...")
def run_ml_pipeline_cached(uploaded_file_obj, target_col, task_type):
    """
    Encapsulates the entire ML pipeline for caching.
    Takes uploaded_file_obj (the Streamlit UploadedFile object), target_col, task_type.
    Returns best_model, best_metric, optimal_test_size, all_results, X_cols, original_categorical_cols.
    """
    # --- ADDED CHECK FOR EMPTY FILE ---
    if uploaded_file_obj.size == 0:
        st.error("The uploaded CSV file is empty. Please upload a file with data.")
        return None, None, None, pd.DataFrame(), None, None # Return empty DataFrame for results

    # Read the DataFrame from the uploaded file object inside the cached function
    try:
        df_input = pd.read_csv(uploaded_file_obj)
    except pd.errors.EmptyDataError:
        st.error("The uploaded CSV file is empty or contains no data columns. Please upload a valid CSV file.")
        return None, None, None, pd.DataFrame(), None, None
    except Exception as e:
        st.error(f"Error reading CSV file: {e}")
        return None, None, None, pd.DataFrame(), None, None

    df_copy = df_input.copy() # Work on a copy to avoid modifying cached df_input

    # Get the agent instance (already cached globally)
    agent_instance = get_agent_ai() # Access it here

    # Preprocess data (fit scaler and encoder here)
    X, y, X_cols, original_categorical_cols = agent_instance._preprocess_data(df_copy, target_col, task_type, fit_scaler_encoder=True)

    if X is None or y is None:
        return None, None, None, pd.DataFrame(), None, None # Return empty DataFrame for results

    # Define test sizes to iterate for optimal split
    test_sizes = [0.1, 0.15, 0.2, 0.25, 0.3]
    overall_best_model = None
    overall_best_metric = float('-inf')
    optimal_test_size = None
    all_results = []

    # Loop through different test sizes
    for ts in test_sizes:
        try:
            # Split data into training and testing sets
            # Use stratify for classification to maintain class proportions
            X_train, X_test, y_train, y_test = train_test_split(
                X, y, test_size=ts, random_state=42,
                stratify=y if task_type == 'Classification' and y.nunique() > 1 else None
            )

            # Loop through all models for the selected task
            for model_name in agent_instance.models[task_type]:
                # Train model
                if agent_instance.train_model(X_train, y_train, task_type, model_name):
                    # Predict
                    y_pred = agent_instance.predict(X_test, task_type, model_name)

                    # Evaluate
                    metric = agent_instance.evaluate(y_test, y_pred, task_type)
                    all_results.append({
                        'Test Size': ts,
                        'Model': model_name,
                        'Metric': metric
                    })

                    # Update best model if current model performs better
                    if metric > overall_best_metric:
                        overall_best_metric = metric
                        overall_best_model = model_name
                        optimal_test_size = ts
                else:
                    # Log failure if model training fails
                    all_results.append({
                        'Test Size': ts,
                        'Model': model_name,
                        'Metric': float('-inf') # Indicate failure
                    })
        except ValueError as ve:
            st.warning(f"Skipping test_size {ts} due to data split error (e.g., too few samples for stratification or single class in split): {ve}")
            continue
        except Exception as e:
            st.error(f"An unexpected error occurred during training/evaluation for test_size {ts}: {e}")
            continue

    return overall_best_model, overall_best_metric, optimal_test_size, pd.DataFrame(all_results), X_cols, original_categorical_cols


# Streamlit app configuration
st.set_page_config(layout="wide", page_title="AI Agent for ML Tasks")
st.title("🤖 AI Agent for Machine Learning Tasks")

st.markdown("""
This AI Agent helps you perform machine learning tasks by automatically detecting
the problem type (Regression or Classification), pre-processing your data,
training various models, and recommending the best performing algorithm.
You can also get predictions on new data!
""")

# Initialize session state variables for chat history and analysis results
if 'chat_history' not in st.session_state:
    st.session_state.chat_history = []
if 'analysis_results' not in st.session_state:
    st.session_state.analysis_results = {
        'overall_best_model': None,
        'overall_best_metric': None,
        'optimal_test_size': None,
        'all_results_df': pd.DataFrame(),
        'X_cols': None,
        'original_categorical_cols': None,
        'df_loaded_hash': None # To track if a new file is loaded or parameters changed
    }
if 'uploaded_file_id' not in st.session_state:
    st.session_state.uploaded_file_id = None
if 'target_col_id' not in st.session_state:
    st.session_state.target_col_id = None
if 'task_type_id' not in st.session_state:
    st.session_state.task_type_id = None


# Create columns for layout
left_column, center_column, right_column = st.columns([2, 4, 2])

# Initialize df here for display and initial column selection in the left column
df = None
uploaded_file = None # Keep uploaded_file as a local variable for the uploader widget

with left_column:
    st.header("📊 Data & Task Setup")
    uploaded_file = st.file_uploader("📂 Choose a CSV file", type=["csv"])

    current_uploaded_file_id = uploaded_file.file_id if uploaded_file else None

    # Check if a new file is uploaded or if the file has changed
    if current_uploaded_file_id != st.session_state.uploaded_file_id:
        st.session_state.uploaded_file_id = current_uploaded_file_id
        # Clear analysis results if a new file is uploaded
        if st.session_state.uploaded_file_id is not None:
            st.session_state.analysis_results = {
                'overall_best_model': None, 'overall_best_metric': None, 'optimal_test_size': None,
                'all_results_df': pd.DataFrame(), 'X_cols': None, 'original_categorical_cols': None,
                'df_loaded_hash': None # Reset hash
            }
            st.session_state.chat_history = [] # Clear chat history too
            st.info("New file uploaded. Please select target and run analysis.")
            st.rerun() # Rerun to reflect cleared state and allow new selections

    if uploaded_file is not None:
        try:
            # --- IMPORTANT: Added the EmptyDataError catch here for the initial read ---
            df = pd.read_csv(uploaded_file)
            st.success("CSV file loaded successfully!")
            st.dataframe(df.head())

            all_columns = df.columns.tolist()
            target_column_name = st.selectbox("🎯 Select Target Column", all_columns, key="target_col_select")

            current_target_col_id = target_column_name # Simple string is hashable

            # Check if target column changed
            if current_target_col_id != st.session_state.target_col_id:
                st.session_state.target_col_id = current_target_col_id
                # Clear analysis results if target column changes
                st.session_state.analysis_results = {
                    'overall_best_model': None, 'overall_best_metric': None, 'optimal_test_size': None,
                    'all_results_df': pd.DataFrame(), 'X_cols': None, 'original_categorical_cols': None,
                    'df_loaded_hash': None
                }
                st.session_state.chat_history = []
                st.info("Target column changed. Please run analysis.")
                st.rerun()


            if target_column_name:
                # Automatic Task Detection Logic
                y_temp = df[target_column_name]
                detected_task = "Regression" # Default to Regression

                # Heuristic for classification: object dtype or low unique numerical values
                if y_temp.dtype == 'object' or y_temp.dtype == 'category':
                    detected_task = "Classification"
                elif pd.api.types.is_numeric_dtype(y_temp):
                    if y_temp.nunique() <= 20 and all(y_temp.dropna().apply(lambda x: x == int(x))):
                        detected_task = "Classification"

                st.info(f"Detected Task Type: **{detected_task}**")
                selected_task_override = st.radio("Override Task Type?", ["Auto-Detect", "Regression", "Classification"], index=0, key="task_override_radio")

                current_task_type_id = selected_task_override # Simple string is hashable

                # Check if task type changed
                if current_task_type_id != st.session_state.task_type_id:
                    st.session_state.task_type_id = current_task_type_id
                    # Clear analysis results if task type changes
                    st.session_state.analysis_results = {
                        'overall_best_model': None, 'overall_best_metric': None, 'optimal_test_size': None,
                        'all_results_df': pd.DataFrame(), 'X_cols': None, 'original_categorical_cols': None,
                        'df_loaded_hash': None
                    }
                    st.session_state.chat_history = []
                    st.info("Task type changed. Please run analysis.")
                    st.rerun()

                if selected_task_override == "Auto-Detect":
                    selected_task = detected_task
                else:
                    selected_task = selected_task_override
                    st.warning(f"Using user-selected task: **{selected_task}**")

                st.markdown("---")
                st.subheader("⚙️ Training Parameters")
                st.write("The agent will automatically find the best `test_size` between 0.1 and 0.3 for optimal model performance.")
                st.slider("Initial Test Size View (for reference)", min_value=0.1, max_value=0.3, step=0.05, value=0.2, disabled=True)

                # --- Run Analysis Button ---
                if st.button("🚀 Run Analysis"):
                    if uploaded_file is not None and target_column_name and selected_task:
                        # Call the cached ML pipeline function
                        (
                            st.session_state.analysis_results['overall_best_model'],
                            st.session_state.analysis_results['overall_best_metric'],
                            st.session_state.analysis_results['optimal_test_size'],
                            st.session_state.analysis_results['all_results_df'],
                            st.session_state.analysis_results['X_cols'],
                            st.session_state.analysis_results['original_categorical_cols']
                        ) = run_ml_pipeline_cached(uploaded_file, target_column_name, selected_task)
                        st.session_state.analysis_results['df_loaded_hash'] = uploaded_file.file_id # Store hash of current file
                        st.session_state.chat_history = [] # Clear chat history on new analysis run
                        st.rerun() # Rerun to display results
                    else:
                        st.warning("Please ensure a CSV file is uploaded, target column is selected, and task type is determined before running analysis.")


        except pd.errors.EmptyDataError:
            st.error("The uploaded CSV file is empty or contains no data columns. Please upload a valid CSV file.")
            df = None # Ensure df is None if file is empty
        except Exception as e:
            st.error(f"Error loading file or selecting target: {e}")
            df = None # Reset df if there's an error
    else:
        st.info("Upload a CSV file to begin.")

with center_column:
    st.header("🧠 Agent AI Performance")
    # Retrieve results from session state
    overall_best_model = st.session_state.analysis_results['overall_best_model']
    overall_best_metric = st.session_state.analysis_results['overall_best_metric']
    optimal_test_size = st.session_state.analysis_results['optimal_test_size']
    all_results_df = st.session_state.analysis_results['all_results_df']
    # X_cols and original_categorical_cols are also in session_state, but used in prediction section

    # Display analysis results only if they exist in session state
    if overall_best_model:
        st.subheader("🏆 Best Model & Performance")
        st.success(f"**Best Model:** `{overall_best_model}`")
        st.success(f"**Best Metric ({'R2 Score' if selected_task == 'Regression' else 'Accuracy'}):** `{overall_best_metric:.4f}`")
        st.success(f"**Optimal Test Size:** `{optimal_test_size*100:.0f}%`")
        st.markdown(f"The agent recommends using the `{overall_best_model}` model with a `{optimal_test_size*100:.0f}%` test split for your dataset, as it yielded the highest performance.")
        if not all_results_df.empty:
            st.subheader("📊 All Model Results")
            # Display results, sorting by metric
            st.dataframe(all_results_df.sort_values(by='Metric', ascending=False))
    elif uploaded_file is not None and target_column_name and selected_task:
        st.info("Analysis not yet run or results cleared. Click 'Run Analysis' to proceed.")
    else:
        st.info("Upload a CSV file, select a target column, and click 'Run Analysis' to see performance.")


    st.subheader("💬 Chat with Agent AI")

    # Display chat history
    for message in st.session_state.chat_history:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])

    # Chat input and logic
    chat_input = st.text_input("Ask me anything about your analysis (e.g., 'What is the best model?', 'How good is the performance?', 'What about the test size?')", key="chat_input")

    if chat_input:
        st.session_state.chat_history.append({"role": "user", "content": chat_input})
        chat_input_lower = chat_input.lower()
        ai_response = ""

        # Retrieve results from session state for chatbot logic
        overall_best_model_chat = st.session_state.analysis_results['overall_best_model']
        overall_best_metric_chat = st.session_state.analysis_results['overall_best_metric']
        optimal_test_size_chat = st.session_state.analysis_results['optimal_test_size']
        # selected_task is a global variable from the main script flow, which is fine for chat responses

        # Check if analysis results are available before responding
        if overall_best_model_chat is None:
            ai_response = "I need to complete the data analysis first before I can answer questions about models or performance. Please upload your data and click 'Run Analysis'."
        elif "best model" in chat_input_lower:
            ai_response = f"The best model for this task is **{overall_best_model_chat}** with a performance metric of **{overall_best_metric_chat:.4f}** achieved with a **{optimal_test_size_chat*100:.0f}%** test split."
        elif "model performance" in chat_input_lower or "how good" in chat_input_lower:
            metric_type = 'R2 Score' if selected_task == 'Regression' else 'Accuracy'
            ai_response = (
                f"The performance of the **{overall_best_model_chat}** model is **{overall_best_metric_chat:.4f}**.\n"
                f"This means the model achieved a {metric_type} of **{overall_best_metric_chat:.4f}**.\n"
            )
            if selected_task == 'Regression':
                ai_response += f"An R2 score of {overall_best_metric_chat*100:.2f}% indicates that this percentage of the variance in the target variable can be explained by the model."
            else:
                ai_response += f"An accuracy of {overall_best_metric_chat*100:.2f}% means the model correctly classified this percentage of samples."

            if overall_best_metric_chat >= 0.8:
                ai_response += "\nThis is a **very good** performance! Your model is highly accurate/predictive."
            elif overall_best_metric_chat >= 0.6:
                ai_response += "\nThis is a **good** performance. There might be room for further improvement with more advanced tuning or feature engineering."
            else:
                ai_response += "\nThe performance is moderate. You might consider more data, different features, or deeper model tuning."
        elif "test size" in chat_input_lower or "split" in chat_input_lower:
            ai_response = f"The optimal test size found for your dataset is **{optimal_test_size_chat*100:.0f}%**. This split provided the best balance for evaluating the models and achieving the highest performance."
        elif "hello" in chat_input_lower or "hi" in chat_input_lower:
            ai_response = "Hello there! How can I assist you with your machine learning task today?"
        else:
            ai_response = f"I'm happy to chat with you about data science! You said: '{chat_input}'. Try asking about 'best model', 'model performance', or 'test size'."

        st.session_state.chat_history.append({"role": "assistant", "content": ai_response}) # Changed role to 'assistant'
        st.rerun() # Rerun to display the new message and clear the input


with right_column:
    st.header("🚀 Make a Prediction")
    # Retrieve results from session state
    overall_best_model = st.session_state.analysis_results['overall_best_model']
    X_cols = st.session_state.analysis_results['X_cols']
    original_categorical_cols = st.session_state.analysis_results['original_categorical_cols']

    # Only show prediction section if analysis results are available AND df is loaded (for feature names)
    if df is not None and target_column_name and selected_task and overall_best_model and X_cols is not None and original_categorical_cols is not None:
        st.markdown(f"Using the best model: **`{overall_best_model}`**")
        st.markdown("---")
        st.subheader("Manual Input for Prediction")

        st.info("Please enter values for the original features. The agent will preprocess them automatically.")

        manual_input_values = {}
        # Get original feature columns (excluding target)
        original_features = [col for col in df.columns if col != target_column_name]

        # Create input fields for original features
        for col in original_features:
            original_col_dtype = df[col].dtype
            if pd.api.types.is_numeric_dtype(original_col_dtype):
                # For numerical columns, use number_input
                manual_input_values[col] = st.number_input(
                    f"Enter value for '{col}' (Numerical)",
                    value=float(df[col].mean()) if not df[col].isnull().all() else 0.0, # Default to mean if available
                    key=f"manual_input_{col}"
                )
            elif original_col_dtype == 'object' or original_col_dtype == 'category':
                # For categorical columns, use selectbox with unique values
                unique_vals = df[col].dropna().unique().tolist()
                if len(unique_vals) > 0:
                    manual_input_values[col] = st.selectbox(
                        f"Select value for '{col}' (Categorical)",
                        [''] + unique_vals, # Add empty string for no selection
                        key=f"manual_input_{col}_cat"
                    )
                else:
                    manual_input_values[col] = st.text_input(
                        f"Enter value for '{col}' (Categorical)",
                        key=f"manual_input_{col}_text"
                    )
            else:
                manual_input_values[col] = st.text_input(
                    f"Enter value for '{col}'",
                    key=f"manual_input_{col}_generic"
                )

        predict_button = st.button("Predict")

        if predict_button:
            # Create a DataFrame from manual input
            manual_input_df = pd.DataFrame([manual_input_values])

            # Apply the same preprocessing steps to the manual input
            # This requires careful handling of columns, especially for one-hot encoding
            # Ensure all original categorical columns are present for consistent dummy creation
            for cat_col in original_categorical_cols:
                if cat_col not in manual_input_df.columns:
                    manual_input_df[cat_col] = np.nan # Add missing categorical columns

            # Fill NaNs in manual input for preprocessing consistency
            for col in manual_input_df.select_dtypes(include=np.number).columns:
                manual_input_df[col] = manual_input_df[col].fillna(0) # Or use mean from training data
            for col in manual_input_df.select_dtypes(include='object').columns:
                manual_input_df[col] = manual_input_df[col].fillna('') # Or use mode from training data

            # Convert categorical columns to 'category' dtype before get_dummies
            for col in original_categorical_cols:
                if col in manual_input_df.columns:
                    manual_input_df[col] = manual_input_df[col].astype('category')

            manual_X_processed = pd.get_dummies(manual_input_df, columns=original_categorical_cols, drop_first=True)

            # Align columns: add missing columns (from training X) and reorder
            missing_cols = set(X_cols) - set(manual_X_processed.columns)
            for c in missing_cols:
                manual_X_processed[c] = 0 # Add columns that were created during training but not present in this single input
            manual_X_processed = manual_X_processed[X_cols] # Ensure order and presence of all columns

            # Scale numerical features using the *fitted* scaler from training data
            numerical_cols_for_scaling = manual_X_processed.select_dtypes(include=np.number).columns
            if len(numerical_cols_for_scaling) > 0:
                manual_X_processed[numerical_cols_for_scaling] = agent.scaler.transform(manual_X_processed[numerical_cols_for_scaling])

            try:
                y_pred_manual = agent.predict(manual_X_processed, selected_task, overall_best_model)
                if y_pred_manual is not None:
                    if selected_task == 'Regression':
                        st.success(f"**Predicted Value:** `{y_pred_manual[0]:.4f}`")
                    else:
                        # Decode the predicted class if target was encoded
                        predicted_class_label = agent.label_encoder.inverse_transform(y_pred_manual)[0]
                        st.success(f"**Predicted Class:** `{predicted_class_label}`")
                else:
                    st.error("Prediction failed. Please check your input and the model.")
            except Exception as e:
                st.error(f"Error during manual prediction: {e}")
    else:
        st.info("Please load data, select a target, and click 'Run Analysis' to enable manual prediction.")
