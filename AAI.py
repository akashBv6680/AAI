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

class AgentAI:
    def __init__(self):
        self.scaler = StandardScaler()
        self.label_encoder = LabelEncoder()
        self.trained_models = {}

        self.models = {
            'Regression': {
                'Linear Regression': LinearRegression(),
                'Polynomial Regression': LinearRegression(),
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
                'Logistic Regression': LogisticRegression(max_iter=2000, random_state=42),
                'Decision Tree Classifier': DecisionTreeClassifier(random_state=42),
                'Random Forest Classifier': RandomForestClassifier(random_state=42),
                'Extra Trees Classifier': ExtraTreesClassifier(random_state=42),
                'Gradient Boosting Classifier': GradientBoostingClassifier(random_state=42),
                'XGBoost Classifier': xgb.XGBClassifier(use_label_encoder=False, eval_metric='logloss', random_state=42),
                'KNN Classifier': KNeighborsClassifier(),
                'SVC': SVC(max_iter=2000, random_state=42),
                'Gaussian Naive Bayes': GaussianNB()
            }
        }

    def _preprocess_data(self, df, target_column, task, fit_scaler_encoder=True):
        df_processed = df.copy()

        y = df_processed[target_column]
        X = df_processed.drop(columns=[target_column])

        self.original_categorical_cols = X.select_dtypes(include='object').columns.tolist()

        numerical_cols = X.select_dtypes(include=np.number).columns
        for col in numerical_cols:
            if X[col].isnull().any():
                X[col] = X[col].fillna(X[col].mean())

        categorical_cols = X.select_dtypes(include='object').columns
        for col in categorical_cols:
            if X[col].isnull().any():
                X[col] = X[col].fillna(X[col].mode()[0])

        for col in self.original_categorical_cols:
            if col in X.columns:
                X[col] = X[col].astype('category')

        X = pd.get_dummies(X, columns=self.original_categorical_cols, drop_first=True)

        numerical_cols_after_dummies = X.select_dtypes(include=np.number).columns
        if len(numerical_cols_after_dummies) > 0:
            if fit_scaler_encoder:
                X[numerical_cols_after_dummies] = self.scaler.fit_transform(X[numerical_cols_after_dummies])
            else:
                X[numerical_cols_after_dummies] = self.scaler.transform(X[numerical_cols_after_dummies])

        if task == 'Classification' and y.dtype == 'object':
            if fit_scaler_encoder:
                y = self.label_encoder.fit_transform(y)
                st.info(f"Target variable encoded. Original classes: {self.label_encoder.classes_}")

        return X, y, X.columns.tolist(), self.original_categorical_cols

    def train_model(self, X_train, y_train, task, model_name, degree=2):
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
        if y_pred is None:
            return -np.inf
        try:
            if task == 'Regression':
                return r2_score(y_test, y_pred)
            else:
                return accuracy_score(y_test, y_pred)
        except Exception as e:
            st.error(f"Error evaluating model: {e}")
            return -np.inf

agent = AgentAI()

st.set_page_config(layout="wide", page_title="AI Agent for ML Tasks")
st.title("🤖 AI Agent for Machine Learning Tasks")

if 'analysis_results' not in st.session_state:
    st.session_state.analysis_results = None
if 'chat_history' not in st.session_state:
    st.session_state.chat_history = []

left_column, center_column, right_column = st.columns([2, 4, 2])

df = None
target_column_name = None
selected_task = None
X_cols = None
original_categorical_cols = []

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
                y_temp = df[target_column_name]
                detected_task = "Regression"

                if y_temp.dtype == 'object' or y_temp.dtype == 'category':
                    detected_task = "Classification"
                elif pd.api.types.is_numeric_dtype(y_temp):
                    if y_temp.nunique() <= 20 and all(y_temp.dropna().apply(lambda x: x == 1)):
                        detected_task = "Classification"

                selected_task_override = st.radio("Override Task Type?", ["Auto-Detect", "Regression", "Classification"], index=0)

                if selected_task_override == "Auto-Detect":
                    selected_task = detected_task
                else:
                    selected_task = selected_task_override
                    st.warning(f"Using user-selected task: **{selected_task}**")

                if st.button("Run Analysis"):
                    X, y, X_cols, original_categorical_cols = agent._preprocess_data(df, target_column_name, selected_task, fit_scaler_encoder=True)

                    if X is not None and y is not None:
                        test_sizes = [0.1, 0.15, 0.2, 0.25, 0.3]
                        overall_best_model = None
                        overall_best_metric = float('-inf')
                        optimal_test_size = None
                        all_results = []

                        for ts in test_sizes:
                            try:
                                X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=ts, random_state=42, stratify=y if selected_task == 'Classification' and y.nunique() > 1 else None)

                                for model_name in agent.models[selected_task]:
                                    if agent.train_model(X_train, y_train, selected_task, model_name):
                                        y_pred = agent.predict(X_test, selected_task, model_name)
                                        metric = agent.evaluate(y_test, y_pred, selected_task)
                                        all_results.append({
                                            'Test Size': ts,
                                            'Model': model_name,
                                            'Metric': metric
                                        })

                                        if metric > overall_best_metric:
                                            overall_best_metric = metric
                                            overall_best_model = model_name
                                            optimal_test_size = ts
                                    else:
                                        all_results.append({
                                            'Test Size': ts,
                                            'Model': model_name,
                                            'Metric': float('-inf')
                                        })
                            except ValueError as ve:
                                st.warning(f"Skipping test_size {ts} due to data split error: {ve}")
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
            df = None

with center_column:
    st.header("🧠 Agent AI Performance")
    if st.session_state.analysis_results:
        overall_best_model = st.session_state.analysis_results.get('overall_best_model')
        overall_best_metric = st.session_state.analysis_results.get('overall_best_metric')
        optimal_test_size = st.session_state.analysis_results.get('optimal_test_size')
        all_results = st.session_state.analysis_results.get('all_results')
        X_cols = st.session_state.analysis_results.get('X_cols')
        original_categorical_cols = st.session_state.analysis_results.get('original_categorical_cols')

        if overall_best_model:
            st.success(f"**Best Model:** `{overall_best_model}`")
            if overall_best_metric is not None:
                st.success(f"**Best Metric ({'R2 Score' if selected_task == 'Regression' else 'Accuracy'}):** `{overall_best_metric:.4f}`")
            if optimal_test_size is not None:
                st.success(f"**Optimal Test Size:** `{optimal_test_size*100:.0f}%`")
            st.markdown(f"The agent recommends using the `{overall_best_model}` model with a `{optimal_test_size*100:.0f}%` test split for your dataset, as it yielded the highest performance.")

with right_column:
    st.header("🚀 Make a Prediction")
    if st.session_state.analysis_results:
        overall_best_model = st.session_state.analysis_results.get('overall_best_model')
        X_cols = st.session_state.analysis_results.get('X_cols')
        original_categorical_cols = st.session_state.analysis_results.get('original_categorical_cols')

        if overall_best_model and X_cols and original_categorical_cols:
            st.markdown(f"Using the best model: **`{overall_best_model}`**")
            st.markdown("---")
            st.subheader("Manual Input for Prediction")

            manual_input_values = {}
            original_features = [col for col in df.columns if col != target_column_name]

            for col in original_features:
                original_col_dtype = df[col].dtype
                if pd.api.types.is_numeric_dtype(original_col_dtype):
                    manual_input_values[col] = st.number_input(
                        f"Enter value for '{col}' (Numerical)",
                        value=float(df[col].mean()) if not df[col].isnull().all() else 0.0
                    )
                elif original_col_dtype == 'object' or original_col_dtype == 'category':
                    unique_vals = df[col].dropna().unique().tolist()
                    if len(unique_vals) > 0:
                        manual_input_values[col] = st.selectbox(
                            f"Select value for '{col}' (Categorical)",
                            [''] + unique_vals
                        )
                    else:
                        manual_input_values[col] = st.text_input(
                            f"Enter value for '{col}' (Categorical)"
                        )
                else:
                    manual_input_values[col] = st.text_input(
                        f"Enter value for '{col}'"
                    )

            predict_button = st.button("Predict")

            if predict_button:
                manual_input_df = pd.DataFrame([manual_input_values])

                for cat_col in original_categorical_cols:
                    if cat_col not in manual_input_df.columns:
                        manual_input_df[cat_col] = np.nan

                for col in manual_input_df.select_dtypes(include=np.number).columns:
                    manual_input_df[col] = manual_input_df[col].fillna(0)
                for col in manual_input_df.select_dtypes(include='object').columns:
                    manual_input_df[col] = manual_input_df[col].fillna('')

                for col in original_categorical_cols:
                    if col in manual_input_df.columns:
                        manual_input_df[col] = manual_input_df[col].astype('category')

                manual_X_processed = pd.get_dummies(manual_input_df, columns=original_categorical_cols, drop_first=True)

                missing_cols = set(X_cols) - set(manual_X_processed.columns)
                for c in missing_cols:
                    manual_X_processed[c] = 0
                manual_X_processed = manual_X_processed[X_cols]

                numerical_cols_for_scaling = manual_X_processed.select_dtypes(include=np.number).columns
                if len(numerical_cols_for_scaling) > 0:
                    manual_X_processed[numerical_cols_for_scaling] = agent.scaler.transform(manual_X_processed[numerical_cols_for_scaling])

                try:
                    y_pred_manual = agent.predict(manual_X_processed, selected_task, overall_best_model)
                    if y_pred_manual is not None:
                        if selected_task == 'Regression':
                            st.success(f"**Predicted Value:** `{y_pred_manual[0]:.4f}`")
                        else:
                            predicted_class_label = agent.label_encoder.inverse_transform(y_pred_manual)[0]
                            st.success(f"**Predicted Class:** `{predicted_class_label}`")
                    else:
                        st.error("Prediction failed. Please check your input and the model.")
                except Exception as e:
                    st.error(f"Error during manual prediction: {e}")
