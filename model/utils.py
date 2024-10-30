import json

import pandas as pd
import numpy as np

from collections import Counter
from scipy.stats import skew, kurtosis
from xgboost import XGBClassifier
from tqdm import tqdm 
import joblib

from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import (
    train_test_split,
    RandomizedSearchCV,
    StratifiedKFold,
)

from concurrent.futures import ProcessPoolExecutor, as_completed


def read_labels(labels_file):
    """
    Read file containing labels for training.

    Args:
        labels_file (str): Path to file containing labels.

    Returns:
        labels_df (pd.DataFrame): Dataframe containing labels.
    """
    labels_df = pd.read_csv(labels_file, delimiter=",")

    return labels_df


def read_data_json(data_json):
    """
    Read data json.

    Args:
        data_json (str): Path to data json file.

    Returns:
        data_list (list): List containing data info.
    """
    # Open the file and read line by line (each line = one read)
    with open(data_json, "r") as file:
        data_list = []
        for line in file:
            try:
                data = json.loads(line)  # Parse each line as a separate JSON object
                data_list.append(data)
            except json.JSONDecodeError as e:
                print(f"Error decoding JSON on line: {line}")
                print(f"Error message: {e}")

    return data_list


def compute_sequence_proportions(data_list, start_idx, stop_idx):
    """
    Takes in the data list and produces proportions of the occurence of the sequence

    Args:
        data_list (list): List containing data info.
        start_idx (int): Start index of cropped sequence. If referring to first index, use 0.
        end_idx (int): End index of cropped sequence. E.g. if stopping at second index, use 2.

    Returns:
        sequence_porportions (dict): Dictionary containing cropped sequence as key and proportion as value.
    """
    sequence_counter = Counter()
    total_sequences = 0

    # First pass: Count the occurrences of each sequence
    for data in data_list:
        for transcript_id, positions in data.items():
            for position, sequence_data in positions.items():
                for sequence in sequence_data.keys():
                    cropped_sequence = sequence[start_idx:stop_idx]
                    sequence_counter[cropped_sequence] += 1
                    total_sequences += 1

    # Compute the proportion for each sequences
    sequence_proportions = {
        seq: count / total_sequences for seq, count in sequence_counter.items()
    }
    return sequence_proportions


def get_5_mer_sequence_proportions(data_list):
    """
    Compute sequence proportions for the 5-mer sequences out of the 7-mer sequence in data_list.

    Args:
        data_list (list): List containing data info.

    Returns:
        sequence_proportions_1 (dict): Dict containing sequence proportion for first 5-mer.
        sequence_proportions_2 (dict): Dict containing sequence proportion for second 5-mer.
        sequence_proportions_3 (dict): Dict containing sequence proportion for third 5-mer.
    """

    start_idx = 0
    end_idx = 5

    sequence_proportions_1 = compute_sequence_proportions(data_list, start_idx, end_idx)
    sequence_proportions_2 = compute_sequence_proportions(
        data_list, start_idx + 1, end_idx + 1
    )
    sequence_proportions_3 = compute_sequence_proportions(
        data_list, start_idx + 2, end_idx + 2
    )

    return sequence_proportions_1, sequence_proportions_2, sequence_proportions_3

def process_transcript_data(
    data,
    start_idx,
    end_idx,
    sequence_proportions_1,
    sequence_proportions_2,
    sequence_proportions_3,
):
    """
    Perform feature engineering on data_list and collate features to create rows, returning to parallel processor to combine to df.

    Args:
        data_list (list): List containing data info.
        start_idx (int): Start index of cropped sequence. If referring to first index, use 0.
        end_idx (int): End index of cropped sequence. E.g. if stopping at second index, use 2.
        sequence_proportions_1 (dict): Dict containing sequence proportion for first 5-mer.
        sequence_proportions_2 (dict): Dict containing sequence proportion for second 5-mer.
        sequence_proportions_3 (dict): Dict containing sequence proportion for third 5-mer.

    Returns:
        rows (List): List containing 85 features for each transcript id and transcript position.

    """
    rows = []
    for transcript_id, positions in data.items():
        for position, sequence_data in positions.items():
            for sequence, measurements in sequence_data.items():
                # Convert to numpy array for easier aggregation
                scores_array = np.array(measurements)

                # Calculate the mean, variance, max, and min along the rows
                mean_scores = np.mean(scores_array, axis=0)
                var_scores = np.var(scores_array, axis=0)
                max_scores = np.max(scores_array, axis=0)
                min_scores = np.min(scores_array, axis=0)
                skew_scores = skew(scores_array, axis=0)
                kurtosis_scores = kurtosis(scores_array, axis=0)
                sequence_1 = sequence[start_idx:end_idx]
                sequence_2 = sequence[start_idx + 1 : end_idx + 1]
                sequence_3 = sequence[start_idx + 2 : end_idx + 2]

                measurement_diff_mean_dwell_one = np.mean(
                    scores_array[:, 3] - scores_array[:, 0]
                )
                measurement_diff_mean_sd_one = np.mean(
                    scores_array[:, 4] - scores_array[:, 1]
                )
                measurement_diff_mean_mean_one = np.mean(
                    scores_array[:, 5] - scores_array[:, 2]
                )

                measurement_diff_mean_dwell_two = np.mean(
                    scores_array[:, 6] - scores_array[:, 3]
                )
                measurement_diff_mean_sd_two = np.mean(
                    scores_array[:, 7] - scores_array[:, 4]
                )
                measurement_diff_mean_mean_two = np.mean(
                    scores_array[:, 8] - scores_array[:, 5]
                )
                transcript_position = {
                    "transcript_id": transcript_id,
                    "transcript_position": position,
                    "sequence": sequence, 
                    "proportion_1": sequence_proportions_1[sequence_1],
                    "proportion_2": sequence_proportions_2[sequence_2],
                    "proportion_3": sequence_proportions_3[sequence_3],
                    "pos": int(position),
                    "diff_1_1": measurement_diff_mean_dwell_one,
                    "diff_1_2": measurement_diff_mean_sd_one,
                    "diff_1_3": measurement_diff_mean_mean_one,
                    "diff_2_1": measurement_diff_mean_dwell_two,
                    "diff_2_2": measurement_diff_mean_sd_two,
                    "diff_2_3": measurement_diff_mean_mean_two,
                    "length": len(measurements),
                }

                for idx in range(scores_array.shape[1]):
                    transcript_position.update(
                        {
                            f"mean_{idx}": mean_scores[idx],
                            f"var_{idx}": var_scores[idx],
                            f"max_{idx}": max_scores[idx],
                            f"min_{idx}": min_scores[idx],
                            f"skewness_{idx}": skew_scores[idx],
                            f"kurtosis_{idx}": kurtosis_scores[idx],
                        }
                    )
                for i in range(3):
                    transcript_position.update(
                        {
                            f"mean_{i}c": mean_scores[i]
                            + mean_scores[i + 3]
                            + mean_scores[i + 6],
                            f"var_{i}c": var_scores[i]
                            + var_scores[i + 3]
                            + var_scores[i + 6],
                            f"max_{i}c": max_scores[i]
                            + max_scores[i + 3]
                            + max_scores[i + 6],
                            f"min_{i}c": min_scores[i]
                            + min_scores[i + 3]
                            + min_scores[i + 6],
                            f"skewness_{i}c": skew_scores[i]
                            + skew_scores[i + 3]
                            + skew_scores[i + 6],
                            f"kurtosis_{i}c": kurtosis_scores[i]
                            + kurtosis_scores[i + 3]
                            + skew_scores[i + 6],
                        }
                    )

                rows.append(transcript_position)

    return rows

def create_aggregated_dataframe(data_list, start_idx, end_idx, sequence_proportions_1, sequence_proportions_2, sequence_proportions_3):
    """
    Handles creation of feature engineered df with paralell processing
    """
    rows = []
    with ProcessPoolExecutor() as executor:
        futures = [
            executor.submit(
                process_transcript_data, data, start_idx, end_idx, sequence_proportions_1, sequence_proportions_2, sequence_proportions_3
            )
            for data in data_list
        ]

        for future in tqdm(as_completed(futures), total=len(data_list)):
            rows.extend(future.result())

    # Create DataFrame
    df = pd.DataFrame(rows)
    return df


def get_full_dataframe(aggregated_df, labels_df):
    """
    Left join dataframe containing features with data containing labels to form full dataframe for splitting and training.

    Args:
        aggregated_df (pd.DataFrame): Dataframe containing features after feature engineering.
        labels_df (pd.DataFrame): Dataframe containing labels.

    Returns:
        df_full (pd.DataFrame): DataFrame containing features and labels combined.
    """

    df_full = pd.merge(
        aggregated_df,
        labels_df[["gene_id", "transcript_id", "transcript_position", "label"]],
        on=["transcript_id", "transcript_position"],
        how="left",
    )

    return df_full


def get_X_Y(df_full):
    """
    Get X and Y datasets from full dataframe.

    Args:
        df_full (pd.DataFrame): DataFrame containing features and labels combined.

    Returns:
        X (pd.DataFrame): Dataframe containing features.
        y (pd.DataFrame): Dataframe containing labels.
    """

    df_final = df_full.copy()

    X = df_final.drop(["label"], axis=1)
    y = df_final[["label", "transcript_id", "transcript_position", "gene_id"]]

    return X, y


def train_test_split_by_geneid(X, y):
    """
    Perform train test split based on gene ids.

    Args:
        X (pd.DataFrame): Dataframe containing features.
        y (pd.DataFrame): Dataframe containing labels.

    Returns:
        X_train (pd.DataFrame): Dataframe containing features for training.
        X_test (pd.DataFrame): Dataframe containing features for testing.
        y_train (pd.DataFrame): Dataframe containing labels for training.
        y_test (pd.DataFrame): Dataframe containing labels for testing.
    """

    unique_gene_ids = X["gene_id"].unique()

    train_gene_ids, test_gene_ids = train_test_split(
        unique_gene_ids, test_size=0.1, random_state=42
    )

    # Create train and test sets by filtering based on gene_id
    X_train = X[X["gene_id"].isin(train_gene_ids)]
    X_test = X[X["gene_id"].isin(test_gene_ids)]

    y_train = y[y["gene_id"].isin(train_gene_ids)]
    y_test = y[y["gene_id"].isin(test_gene_ids)]

    # Drop unnecessary columns
    X_train = X_train.drop(["transcript_id", "transcript_position", "sequence", "gene_id"], axis=1)
    X_test = X_test.drop(["transcript_id", "transcript_position", "sequence", "gene_id"], axis=1)
    y_train = y_train["label"]
    y_test = y_test["label"]

    return X_train, X_test, y_train, y_test


def scale_X_features(X_train, X_test):
    """
    Scale X features using StandardScaler.

    Args:
        X_train (pd.DataFrame): Dataframe containing features for training.
        X_test (pd.DataFrame): Dataframe containing features for testing.

    Returns:
        X_train_scaled (pd.DataFrame): Dataframe containing scaled features for training.
        X_test_scaled (pd.DataFrame): Dataframe containing scaled features for testing.
    """

    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    return X_train_scaled, X_test_scaled


def get_best_params(X_train_scaled, y_train):
    """
    Get best parameters for XGBoost.

    Args:
        X_train_scaled (pd.DataFrame): Dataframe containing scaled features for training.
        y_train (pd.DataFrame): Dataframe containing labels for training.

    Returns:
        best_params (dict): Dictionary containing best parameters found.
    """
    # Define the parameter grid
    param_grid = {
        "n_estimators": [1000, 1100, 1200],  # Default = 100
        "max_depth": [11, 12, 13],  # Default = 6
        "learning_rate": [0.01, 0.05, 0.1],
        "subsample": [0.8, 0.9, 1.0],  # Default = 1
        "colsample_bytree": [0.8, 0.9, 1.0],  # Default = 1
        "gamma": [0, 1],  # Default = 0
        "min_child_weight": range(1, 6),
        "max_delta_step": [0, 1, 2],
    }

    xgb_clf = XGBClassifier(random_state=42)

    # Perform Randomized Search
    random_search = RandomizedSearchCV(
        estimator=xgb_clf,
        param_distributions=param_grid,
        n_iter=50,
        scoring="roc_auc",
        cv=StratifiedKFold(n_splits=3),
        random_state=42,
        n_jobs=-1,
        verbose=2,
    )

    random_search.fit(X_train_scaled, y_train)

    best_params = random_search.best_params_

    return best_params


def get_scale_pos_weights(labels):
    """
    Get scale_pos_weights for XGBClassifier based on counts of classes in labels file.

    Args:
        labels (list): List containing labels.

    Returns:
        pos_weight (float): Weighted classes.
    """

    counter = Counter(labels)

    pos_weights = counter[0] / counter[1]

    return pos_weights
