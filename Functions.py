# Import Modules
import os
import numpy as np
import logging
import time
import gc
from sklearn.utils import shuffle
import seaborn as sns
from imblearn.pipeline import Pipeline
from joblib import Parallel, delayed
import matplotlib.pyplot as plt
from sklearn.metrics import make_scorer, matthews_corrcoef, accuracy_score, precision_score, recall_score, f1_score
from imblearn.over_sampling import SMOTE, ADASYN
from sklearn.model_selection import train_test_split, cross_validate, StratifiedKFold
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.metrics import accuracy_score, matthews_corrcoef, make_scorer, confusion_matrix, ConfusionMatrixDisplay
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import GaussianNB
from xgboost import XGBClassifier
from memory_profiler import profile


# Function Definitions

def get_logger(file_path, log_file_extension='.log'):
    dataset_name = os.path.splitext(os.path.basename(file_path))[0]
    log_file_path = f"{dataset_name}{log_file_extension}"
    
    logger = logging.getLogger(dataset_name)
    logger.setLevel(logging.INFO)
    
    if not logger.hasHandlers():
        console_handler = logging.StreamHandler()
        file_handler = logging.FileHandler(log_file_path)
        
        formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
        console_handler.setFormatter(formatter)
        file_handler.setFormatter(formatter)
        
        logger.addHandler(console_handler)
        logger.addHandler(file_handler)
    
    return logger, dataset_name

def process_chunk(chunk, logger):
    try:
        logger.info("Processing a chunk of data.")
        
        # Handle missing values
        chunk = chunk.apply(lambda col: col.fillna(col.mode()[0]) if col.dtype == 'object' else col.fillna(col.mean()))
        
        # Identify and encode categorical features
        categorical_features = chunk.select_dtypes(include=['object']).columns
        if not categorical_features.empty:
            logger.info(f"Encoding categorical features: {categorical_features.tolist()}")
            for column in categorical_features:
                chunk[column] = LabelEncoder().fit_transform(chunk[column])
        
        return chunk
    except Exception as e:
        logger.error(f"Error processing chunk: {e}")
        raise

def clean_label(file_name):
    # Remove the file extension
    cleaned_name = file_name.rsplit('.', 1)[0]
    # Split by periods and remove the first part if it is a digit
    parts = cleaned_name.split('.')
    if parts[0].isdigit():
        parts.pop(0)
    # Join the remaining parts to form the label
    cleaned_name = '.'.join(parts)
    return cleaned_name

def drop_duplicates_and_shuffle(df):
    df.drop_duplicates(inplace=True)
    df = shuffle(df).reset_index(drop=True)
    return df

def drop_columns_with_zeros_or_nulls(df):
    cols_to_drop = df.columns[(df == 0).all() | df.isna().all()]
    df.drop(columns=cols_to_drop, inplace=True)
    return df

def downcast_dtypes(df, logger):
    try:
        logger.info("Downcasting data types for memory optimization.")
        start_mem = df.memory_usage().sum() / 1024**2
        logger.info(f"Memory usage before downcasting: {start_mem:.2f} MB")
        
        float_cols = [c for c in df if df[c].dtype == "float64"]
        int_cols = [c for c in df if df[c].dtype in ["int64", "int32"]]
        
        df[float_cols] = df[float_cols].astype(np.float32)
        df[int_cols] = df[int_cols].astype(np.int32)
        
        end_mem = df.memory_usage().sum() / 1024**2
        logger.info(f"Memory usage after downcasting: {end_mem:.2f} MB")
        logger.info(f"Memory reduction: {100 * (start_mem - end_mem) / start_mem:.1f}%")
        
        return df
    except Exception as e:
        logger.error(f"Error during downcasting: {e}")
        raise

def reduce_float_precision(df, decimals, logger):
    try:
        logger.info(f"Reducing float precision to {decimals} decimals.")
        float_cols = df.select_dtypes(include=['float32', 'float64']).columns
        df[float_cols] = df[float_cols].round(decimals)
        return df
    except Exception as e:
        logger.error(f"Error reducing float precision: {e}")
        raise

def check_and_handle_inf_values(df, replacement_value=np.nan, logger=None):
     # Check for inf values in the DataFrame
    inf_mask = df.isin([np.inf, -np.inf])

    # Log information about inf values if logger is provided
    if logger:
        if inf_mask.any().any():
            inf_columns = df.columns[inf_mask.any()]
            inf_rows = df.index[inf_mask.any(axis=1)]
            logger.warning(f"Inf values found in columns: {inf_columns.tolist()}")
            logger.warning(f"Inf values found in rows: {inf_rows.tolist()}")
        else:
            logger.info("No inf values found in the DataFrame.")

    # Replace inf values with the specified replacement value
    df.replace([np.inf, -np.inf], replacement_value, inplace=True)

    # Optionally log the replacement action
    if logger:
        logger.info(f"Inf values replaced with {replacement_value}.")

    return df

def handle_missing_values(df, method='drop', axis=0, fill_value=None, strategy=None):
    """
   - axis (int): Axis along which to drop missing values (0 for rows, 1 for columns).
    """
    if method == 'drop':
        df_cleaned = df.dropna(axis=axis)
    elif method == 'fill':
        if strategy == 'mean':
            df_cleaned = df.fillna(df.mean())
        elif strategy == 'median':
            df_cleaned = df.fillna(df.median())
        elif strategy == 'mode':
            df_cleaned = df.fillna(df.mode().iloc[0])
        elif fill_value is not None:
            df_cleaned = df.fillna(fill_value)
        else:
            raise ValueError("For 'fill' method, either 'strategy' or 'fill_value' must be provided.")
    else:
        raise ValueError("Method must be either 'drop' or 'fill'.")

    return df_cleaned

def normalize_data(df, logger):
    try:
        logger.info("Normalizing data...")
        scaler = StandardScaler()
        numerical_features = df.select_dtypes(include=[np.number]).columns
        df[numerical_features] = scaler.fit_transform(df[numerical_features])
        logger.info("Normalization complete.")
        return df
    except Exception as e:
        logger.error(f"Error during normalization: {e}")
        raise

def plot_and_save_correlation_matrix(df, file_path, logger, dpi):
    try:
        logger.info("Plotting and saving the correlation matrix.")
        dataset_name = os.path.splitext(os.path.basename(file_path))[0]
        corr = df.corr()
        plt.figure(figsize=(12, 10))
        sns.heatmap(corr, annot=False, fmt=".2f", cmap='coolwarm', cbar=True, center=0,
                    annot_kws={"size": 5}, linewidths=0.5, cbar_kws={"shrink": 0.75}
                    )
        plt.xticks(rotation= 45, ha='right', fontsize=8)
        plt.yticks(fontsize=8)
        plt.title(f'Correlation coefficient Matrix for all features of {dataset_name}')
        corr_matrix_file = f"{dataset_name}_correlation_full.png"
        # Save the figure with high DPI for better quality
        plt.savefig(corr_matrix_file, dpi=dpi)
        logger.info(f"Correlation matrix saved as {corr_matrix_file} with DPI={dpi}.")
        plt.close()
    except Exception as e:
        logger.error(f"Error while plotting or saving the correlation matrix: {e}")
        raise

def split_data(df, class_label, test_size=0.2, random_state=42, logger=None):
    try:
        logger.info("Splitting data into training and testing sets.")
        
        X = df.drop(columns=[class_label])
        X = normalize_data(X, logger) #normalize dataset excluding the class label
        y = df[class_label]
        
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, random_state=random_state, stratify=y)
        
        train_df = X_train.copy()
        train_df[class_label] = y_train
        
        logger.info(f"Data split complete. Training data shape: {X_train.shape}, Testing data shape: {X_test.shape}")
        return train_df, X_test, y_test
    except Exception as e:
        logger.error(f"Error during data splitting: {e}")
        raise
    
def create_subsets(df, class_label, num_subsets, logger=None):
    try:
        columns = list(df.columns)
        if class_label in columns:
            columns.remove(class_label)
        else:
            if logger:
                logger.error(f"Class label '{class_label}' not found in DataFrame columns.")
            return []

        columns_per_subset = len(columns) // num_subsets
        remainder = len(columns) % num_subsets

        subsets = []
        start_idx = 0
        for i in range(num_subsets):
            end_idx = start_idx + columns_per_subset + (1 if i < remainder else 0)
            subset_columns = columns[start_idx:end_idx] + [class_label]
            subsets.append((i, df[subset_columns]))
            start_idx = end_idx

        if logger:
            logger.info(f"Created {num_subsets} subsets from the training DataFrame.")
        return subsets
    except Exception as e:
        if logger:
            logger.error(f"Error in creating subsets: {e}")
        raise

def rank_and_evaluate_single_subset(subset_id, subset_df, ranking_func, class_label, n_features, n_iter):
    max_accuracy = 0
    best_features = []
    best_iteration = -1

    X = subset_df.drop(columns=[class_label])
    y = subset_df[class_label]

    ranked_features = ranking_func(X.copy(), y, no_features=len(X.columns)).columns.tolist()

    for iteration in range(n_iter):
        start_index = iteration % len(ranked_features)
        selected_features = [ranked_features[(start_index + i) % len(ranked_features)] for i in range(n_features)]

        X_new = X[selected_features]
        y_new = y.copy()            
        X_train, X_test, y_train, y_test = train_test_split(X_new, y_new, test_size=0.2, random_state=42, stratify=y_new)
        smote = SMOTE(random_state=42)
        X_train_resampled, y_train_resampled = smote.fit_resample(X_train, y_train)

        clf = RandomForestClassifier(n_estimators=100, random_state=42)
        clf.fit(X_train_resampled, y_train_resampled)
        y_pred = clf.predict(X_test)

        accuracy = accuracy_score(y_test, y_pred)
        if accuracy > max_accuracy:
            max_accuracy = accuracy
            best_iteration = iteration + 1
            best_features = selected_features

    return {
        'subset_id': subset_id,
        'best_iteration': best_iteration,
        'max_accuracy': max_accuracy,
        'best_features': best_features
    }
@profile
def parallel_rank_and_evaluate(df_subsets, ranking_func, class_label, n_features, n_iter, n_jobs, logger):
    start_time = time.time()
    results = Parallel(n_jobs=n_jobs)(
        delayed(rank_and_evaluate_single_subset)(subset_id, subset_df, ranking_func, class_label, n_features, n_iter)
        for subset_id, subset_df in df_subsets
    )
    elapsed_time = time.time() - start_time
    logger.info(f"Ranking and evaluation with {ranking_func.__name__} took {elapsed_time:.2f} seconds")
    return results

def get_combined_features(results_list, min_ranker_agreement=2):
    from collections import Counter
    
    # Flatten all features from the results into a single list
    all_features = [feature for results in results_list for result in results for feature in result['best_features']]
    
    # Count occurrences of each feature
    feature_counts = Counter(all_features)
    
    # Select features that meet the threshold of ranker agreement
    combined_features = [feature for feature, count in feature_counts.items() if count >= min_ranker_agreement]
    
    return combined_features

def drop_highly_correlated_features(df, combined_features, threshold=0.8, logger=None):
    # Extract the combined features from the DataFrame
    df_selected = df[combined_features]
    
    # Calculate the correlation matrix
    corr_matrix = df_selected.corr().abs()
    
    # Upper triangle of the correlation matrix (without the diagonal)
    upper_tri = corr_matrix.where(np.triu(np.ones(corr_matrix.shape), k=1).astype(bool))
    
    # Find columns with correlations greater than the threshold
    to_drop = [column for column in upper_tri.columns if any(upper_tri[column] > threshold)]
    
    # Drop the highly correlated features
    reduced_features = [feature for feature in combined_features if feature not in to_drop]
    
    # Count the number of selected features
    total_features = len(df.columns)
    selected_features = len(reduced_features)
    
    if logger:
        logger.info(f"Dropping {len(to_drop)} features due to high correlation.")
        logger.info(f"Features dropped due to high correlation: {to_drop}")
        logger.info(f"Selected {selected_features} out of {total_features} features.")

    return reduced_features

def display_correlation_of_selected_features(df, reduced_features, dataset_name, dpi=400, logger=None):
    try:
        if logger:
            logger.info(f"Displaying correlation matrix for optimal features from {dataset_name}.")

        # Selecting the reduced features from the DataFrame
        selected_df = df[reduced_features]
        
        # Calculating the correlation matrix
        corr_matrix = selected_df.corr()
        
        # Plotting the correlation matrix
        plt.figure(figsize=(10, 8))
        sns.heatmap(corr_matrix, annot=True, fmt=".2f", cmap='coolwarm', cbar=True, center=0,
                    annot_kws={"size": 7}, linewidths=0.5, cbar_kws={"shrink": 0.75}
                    )
        plt.xticks(rotation= 45, ha='right', fontsize=8)
        plt.yticks(fontsize=8)
        plt.title(f'Correlation Coefficient Matrix for Optimal Features of {dataset_name}')
        
        # Saving the plot as an image file
        plot_filename = f"{dataset_name}_correlation_optimal.png"
        plt.savefig(plot_filename, dpi=dpi)
        
        if logger:
            logger.info(f"Optimal features correlation matrix saved as {plot_filename}")
        
        # Close the plot to free memory
        plt.close()
        
    except Exception as e:
        if logger:
            logger.error(f"Error displaying or saving correlation matrix of optimal features: {e}")
        raise

def false_positive_rate(y_true, y_pred):
    cm = confusion_matrix(y_true, y_pred)
    fp = cm[0, 1]
    tn = cm[0, 0]
    return fp / (fp + tn)

def detection_rate(y_true, y_pred):
    cm = confusion_matrix(y_true, y_pred)
    tp = cm[1, 1]
    fn = cm[1, 0]
    return tp / (tp + fn)

@profile
def cross_validate_model(train_df, X_test, y_test, selected_features, class_label, logger):
    mcc_scorer = make_scorer(matthews_corrcoef)
    fpr_scorer = make_scorer(false_positive_rate)
    dr_scorer = make_scorer(detection_rate)

    X_train = train_df[selected_features]
    y_train = train_df[class_label]
    X_test_selected = X_test[selected_features]

    classifiers = {
        'RandomForest': RandomForestClassifier(random_state=42, class_weight='balanced'),
        'AdaBoost': AdaBoostClassifier(random_state=42, algorithm="SAMME"),
        'DecisionTree': DecisionTreeClassifier(random_state=42),
        'LogisticRegression': LogisticRegression(random_state=42, solver='lbfgs', max_iter=300),
        'NaiveBayes': GaussianNB(),
        'XGBoost': XGBClassifier(random_state=42, eval_metric='logloss')
    }

    for clf_name, clf in classifiers.items():
        pipeline_without_smote = Pipeline([
            ('scaler', StandardScaler()),
            ('classifier', clf)
        ])

        pipeline_with_smote = Pipeline([
            ('smote', SMOTE(random_state=42)),
            ('scaler', StandardScaler()),
            ('classifier', clf)
        ])
        pipeline_with_adasyn = Pipeline([
            ('adysyn', ADASYN(random_state=42)),
            ('scaler', StandardScaler()),
            ('classifier', clf)
        ])
        cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
        scoring = {
            'accuracy': 'accuracy',
            'precision': 'precision_weighted',
            'recall': 'recall_weighted',
            'f1': 'f1_weighted',
            'mcc': mcc_scorer,
            'fpr': fpr_scorer,
            'dr': dr_scorer
        }

        # Cross-validation without sampling
        logger.info(f"Evaluating {clf_name} without sampling...")
        start_time = time.time()
        scores_without_sampling = cross_validate(pipeline_without_smote, X_train, y_train, cv=cv, scoring=scoring, n_jobs=-1, return_estimator=True)
        elapsed_time = time.time() - start_time
        logger.info(f"{clf_name} without sampling - Cross-validation took {elapsed_time:.2f} seconds")

        for metric in scoring.keys():
            logger.info(f'{clf_name} without sampling - {metric.capitalize()}: {scores_without_sampling["test_" + metric].mean():.4f} ± {scores_without_sampling["test_" + metric].std():.4f}')

        # Cross-validation with SMOTE
        logger.info(f"Evaluating {clf_name} with SMOTE...")
        start_time = time.time()
        scores_with_smote = cross_validate(pipeline_with_smote, X_train, y_train, cv=cv, scoring=scoring, n_jobs=-1, return_estimator=True)
        elapsed_time = time.time() - start_time
        logger.info(f"{clf_name} with SMOTE - Cross-validation took {elapsed_time:.2f} seconds")

        for metric in scoring.keys():
            logger.info(f'{clf_name} with SMOTE - {metric.capitalize()}: {scores_with_smote["test_" + metric].mean():.4f} ± {scores_with_smote["test_" + metric].std():.4f}')

        # Cross-validation with ADASYN
        logger.info(f"Evaluating {clf_name} with ADASYN...")
        start_time = time.time()
        scores_with_adasyn = cross_validate(pipeline_with_adasyn, X_train, y_train, cv=cv, scoring=scoring, n_jobs=-1, return_estimator=True)
        elapsed_time = time.time() - start_time
        logger.info(f"{clf_name} with ADASYN - Cross-validation took {elapsed_time:.2f} seconds")

        for metric in scoring.keys():
            logger.info(f'{clf_name} with SMOTE - {metric.capitalize()}: {scores_with_adasyn["test_" + metric].mean():.4f} ± {scores_with_adasyn["test_" + metric].std():.4f}')

        # Evaluate the last estimator without sampling
        last_estimator_without_smote = scores_without_sampling['estimator'][-1]
        y_pred_without_sampling = last_estimator_without_smote.predict(X_test_selected)

        # Confusion Matrix without sampling
        cm_without_sampling = confusion_matrix(y_test, y_pred_without_sampling)
        disp_without_sampling = ConfusionMatrixDisplay(confusion_matrix=cm_without_sampling, display_labels=last_estimator_without_smote.classes_)
        disp_without_sampling.plot(cmap=plt.cm.Blues)
        plt.title(f"Confusion Matrix: {clf_name} without sampling")
        plt.show()

        # Evaluate the last estimator with SMOTE
        last_estimator_with_smote = scores_with_smote['estimator'][-1]
        y_pred_with_smote = last_estimator_with_smote.predict(X_test_selected)
       
        # Confusion Matrix with SMOTE
        cm_with_smote = confusion_matrix(y_test, y_pred_with_smote)
        disp_with_smote = ConfusionMatrixDisplay(confusion_matrix=cm_with_smote, display_labels=last_estimator_with_smote.classes_)
        disp_with_smote.plot(cmap=plt.cm.Blues)
        plt.title(f"Confusion Matrix: {clf_name} with SMOTE")
        plt.show()

        # Evaluate the last estimator with ADASYN
        last_estimator_with_adasyn = scores_with_adasyn['estimator'][-1]
        y_pred_with_adasyn = last_estimator_with_adasyn.predict(X_test_selected)
       
        # Confusion Matrix with ADASYN
        cm_with_adasyn = confusion_matrix(y_test, y_pred_with_adasyn)
        disp_with_adasyn = ConfusionMatrixDisplay(confusion_matrix=cm_with_adasyn, display_labels=last_estimator_with_adasyn.classes_)
        disp_with_adasyn.plot(cmap=plt.cm.Blues)
        plt.title(f"Confusion Matrix: {clf_name} with ADASYN")
        plt.show()


def plot_confusion_matrix(y_true, y_pred, clf_name, sampling):
    """
    Parameters:
    - y_true: True labels.
    - y_pred: Predicted labels.
    - clf_name: Name of the classifier.
    - sampling: Sampling technique used.
    """
    cm = confusion_matrix(y_true, y_pred)
    disp = ConfusionMatrixDisplay(confusion_matrix=cm)
    disp.plot(cmap=plt.cm.Blues)
    plt.title(f"Confusion Matrix: {clf_name} {sampling}")
    plt.show()

def evaluate_classifiers(train_df, X_test, y_test, selected_features, class_label, logger):
    # Prepare training and testing data
    X_train = train_df[selected_features]
    y_train = train_df[class_label]
    X_test_selected = X_test[selected_features]

    # Define classifiers with default settings
    classifiers = {
        'RandomForest': RandomForestClassifier(random_state=42, class_weight='balanced'),
        'AdaBoost': AdaBoostClassifier(random_state=42, algorithm="SAMME"),
        'DecisionTree': DecisionTreeClassifier(random_state=42),
        'LogisticRegression': LogisticRegression(random_state=42, solver='lbfgs', max_iter=300),
        'NaiveBayes': GaussianNB(),
        'XGBoost': XGBClassifier(random_state=42, eval_metric='logloss')
    }

    # Define metrics for evaluation
    scoring_functions = {
        'accuracy': accuracy_score,
        'precision': precision_score,
        'recall': recall_score,
        'f1': f1_score,
        'mcc': matthews_corrcoef
    }
    
    # Function to evaluate and log metrics
    def evaluate_metrics(y_true, y_pred, clf_name, sampling):
        for metric_name, scorer in scoring_functions.items():
            score = scorer(y_true, y_pred, average='weighted') if metric_name in ['precision', 'recall', 'f1'] else scorer(y_true, y_pred)
            logger.info(f'{clf_name} {sampling} - {metric_name.capitalize()}: {score:.4f}')

    # Function to plot confusion matrix
    def plot_confusion_matrix(y_true, y_pred, clf_name, sampling):
        cm = confusion_matrix(y_true, y_pred)
        disp = ConfusionMatrixDisplay(confusion_matrix=cm)
        disp.plot(cmap=plt.cm.Blues)
        plt.title(f"Confusion Matrix: {clf_name} {sampling}")
        plt.show()

    sampling_methods = {
        'without sampling': None,
        'with SMOTE': SMOTE(random_state=42, n_jobs=-1),
        'with ADASYN': ADASYN(random_state=42, n_jobs=-1)
    }

    for clf_name, clf in classifiers.items():
        for sampling_name, sampler in sampling_methods.items():
            logger.info(f"Evaluating {clf_name} {sampling_name}...")

            # Define pipeline
            steps = [('scaler', StandardScaler())]
            if sampler:
                steps.insert(0, ('sampler', sampler))
            steps.append(('classifier', clf))
            pipeline = Pipeline(steps)

            # Fit and predict
            start_time = time.time()
            pipeline.fit(X_train, y_train)
            elapsed_time = time.time() - start_time
            logger.info(f"{clf_name} {sampling_name} - Training took {elapsed_time:.2f} seconds")

            y_pred = pipeline.predict(X_test_selected)
            evaluate_metrics(y_test, y_pred, clf_name, sampling_name)
            plot_confusion_matrix(y_test, y_pred, clf_name, sampling_name)

    # Cleanup to release memory
    del X_train, y_train, X_test_selected
    gc.collect()
