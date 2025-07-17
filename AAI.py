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
# Create an instance of the AI agent
agent = AgentAI()

# Streamlit app configuration
st.set_page_config(layout="wide", page_title="AI Agent for ML Tasks")
st.title("🤖 AI Agent for Machine Learning Tasks")

# Initialize session state variables for analysis results and chat history
if 'analysis_results' not in st.session_state:
    st.session_state.analysis_results = None
if 'chat_history' not in st.session_state:
    st.session_state.chat_history = []

# Create columns for layout
left_column, center_column, right_column = st.columns([2, 4, 2])

df = None
target_column_name = None
selected_task = None # To store the task selected by the user or detected
X_cols = None # To store feature columns after preprocessing for consistent prediction input
original_categorical_cols = [] # To store original categorical columns for manual input preprocessing

with left_column:
    st.header("📊 Data & Task Setup")
    uploaded_file = st.file_uploader("📂 Choose a CSV file", type=["csv"])

    if uploaded_file is not None:
        try:
            df = pd.read_csv(uploaded_file)
            st.success("CSV file loaded successfully!")
            st.dataframe(df.head())

            all_columns = df.columns.tolist()
            target_column_name = st.selectbox("🎯 Select Target Column", all_columns)

            if target_column_name:
                # Automatic Task Detection Logic
                y_temp = df[target_column_name]
                detected_task = "Regression" # Default to Regression

                # Heuristic for classification: object dtype or low unique numerical values
                if y_temp.dtype == 'object' or y_temp.dtype == 'category':
                    detected_task = "Classification"
                elif pd.api.types.is_numeric_dtype(y_temp):
                    # If numerical, check ratio of unique values to total values
                    # and if it's mostly integer-like
                    if y_temp.nunique() <= 20 and all(y_temp.dropna().apply(lambda x: x == int(x))): # Max 20 unique values for classification guess
                        detected_task = "Classification"

                st.info(f"Detected Task Type: **{detected_task}**")
                # User can override the detected task
                selected_task_override = st.radio("Override Task Type?", ["Auto-Detect", "Regression", "Classification"], index=0)

                if selected_task_override == "Auto-Detect":
                    selected_task = detected_task
                else:
                    selected_task = selected_task_override
                    st.warning(f"Using user-selected task: **{selected_task}**")

                st.markdown("---")
                st.subheader("⚙️ Training Parameters")
                st.write("The agent will automatically find the best `test_size` between 0.1 and 0.3 for optimal model performance.")
                st.slider("Initial Test Size View (for reference)", min_value=0.1, max_value=0.3, step=0.05, value=0.2, disabled=True)

                if st.button("Run Analysis"):
                    if uploaded_file is not None and target_column_name and selected_task:
                        X, y, X_cols, original_categorical_cols = agent._preprocess_data(df, target_column_name, selected_task, fit_scaler_encoder=True)

                        if X is not None and y is not None:
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
                                        stratify=y if selected_task == 'Classification' and y.nunique() > 1 else None
                                    )

                                    # Loop through all models for the selected task
                                    for model_name in agent.models[selected_task]:
                                        # Train model
                                        if agent.train_model(X_train, y_train, selected_task, model_name):
                                            # Predict
                                            y_pred = agent.predict(X_test, selected_task, model_name)

                                            # Evaluate
                                            metric = agent.evaluate(y_test, y_pred, selected_task)
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

                            st.session_state.analysis_results = {
                                'overall_best_model': overall_best_model,
                                'overall_best_metric': overall_best_metric,
                                'optimal_test_size': optimal_test_size,
                                'all_results': all_results,
                                'X_cols': X_cols,
                                'original_categorical_cols': original_categorical_cols
                            }

        except Exception as e:
            st.error(f"Error loading file or selecting target: {e}")
            df = None # Reset df if there's an error

with center_column:
    st.header("🧠 Agent AI Performance")
    if st.session_state.analysis_results:
        overall_best_model = st.session_state.analysis_results['overall_best_model']
        overall_best_metric = st.session_state.analysis_results['overall_best_metric']
        optimal_test_size = st.session_state.analysis_results['optimal_test_size']
        all_results = st.session_state.analysis_results['all_results']
        X_cols = st.session_state.analysis_results['X_cols']
        original_categorical_cols = st.session_state.analysis_results['original_categorical_cols']

        st.subheader("🏆 Best Model & Performance")
        if overall_best_model:
            st.success(f"**Best Model:** `{overall_best_model}`")
            st.success(f"**Best Metric ({'R2 Score' if selected_task == 'Regression' else 'Accuracy'}):** `{overall_best_metric:.4f}`")
            st.success(f"**Optimal Test Size:** `{optimal_test_size*100:.0f}%`")
            st.markdown(f"The agent recommends using the `{overall_best_model}` model with a `{optimal_test_size*100:.0f}%` test split for your dataset, as it yielded the highest performance.")

        # Chat interface
        chat_input = st.text_input("Ask me anything about your analysis")
        if chat_input:
            st.session_state.chat_history.append({"role": "You", "content": chat_input})
            # Process chat input and respond
            chat_input_lower = chat_input.lower()
            ai_response = ""

            if "best model" in chat_input_lower:
                if overall_best_model:
                    ai_response = f"The best model for this task is **{overall_best_model}** with a performance metric of **{overall_best_metric:.4f}** achieved with a **{optimal_test_size*100:.0f}%** test split."
                else:
                    ai_response = "I haven't been able to determine the best model yet. Please ensure data is loaded and processed correctly."
            elif "model performance" in chat_input_lower or "how good" in chat_input_lower:
                if overall_best_model:
                    metric_type = 'R2 Score' if selected_task == 'Regression' else 'Accuracy'
                    ai_response = (
                        f"The performance of the **{overall_best_model}** model is **{overall_best_metric:.4f}**.\n"
                        f"This means the model achieved a {metric_type} of **{overall_best_metric:.4f}**.\n"
                    )
                    if selected_task == 'Regression':
                        ai_response += f"An R2 score of {overall_best_metric*100:.2f}% indicates that this percentage of the variance in the target variable can be explained by the model."
                    else:
                        ai_response += f"An accuracy of {overall_best_metric*100:.2f}% means the model correctly classified this percentage of samples."

                    if overall_best_metric >= 0.8:
                        ai_response += "\nThis is a **very good** performance! Your model is highly accurate/predictive."
                    elif overall_best_metric >= 0.6:
                        ai_response += "\nThis is a **good** performance. There might be room for further improvement with more advanced tuning or feature engineering."
                    else:
                        ai_response += "\nThe performance is moderate. You might consider more data, different features, or deeper model tuning."
                else:
                    ai_response = "I need to analyze the data first to tell you about model performance."
            else:
                ai_response = f"I'm happy to chat with you about data science! You said: '{chat_input}'. Try asking about 'best model', 'model performance', or 'test size'."

            st.session_state.chat_history.append({"role": "Agent AI", "content": ai_response})

        for message in st.session_state.chat_history:
            st.write(f"**{message['role']}**: {message['content']}")

with right_column:
    st.header("🚀 Make a Prediction")
    if st.session_state.analysis_results:
        overall_best_model = st.session_state.analysis_results['overall_best_model']
        X_cols = st.session_state.analysis_results['X_cols']
        original_categorical_cols = st.session_state.analysis_results['original_categorical_cols']

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
