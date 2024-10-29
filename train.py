import pickle
import os
import argparse

from xgboost import XGBClassifier

from utils import (
    read_labels,
    read_data_json,
    get_5_mer_sequence_proportions,
    create_aggregated_dataframe,
    get_full_dataframe,
    get_X_Y,
    train_test_split_by_geneid,
    scale_X_features,
    get_scale_pos_weights,
)


def train_model(data_json_path, labels_file_path, output_path, output_filename):
    """
    Trains XGBoost Model and saves trained model to specified path.

    Args:
        data_json_path (str): Path to data json file.
        labels_file_path (str): Path to file containing labels.
        output_path (str): Path to save trained model.
        output_filename (str): Filename to save trained model.

    Returns:
        model: Trained model.
    """

    print("Reading labels...")
    labels_df = read_labels(labels_file_path)
    print("Reading data json...")
    data_list = read_data_json(data_json_path)

    print("Computing sequence proportions...")
    sequence_proportions_1, sequence_proportions_2, sequence_proportions_3 = (
        get_5_mer_sequence_proportions(data_list)
    )

    print("Performing feature engineering...")
    start_idx = 0
    end_idx = 5
    aggregated_df = create_aggregated_dataframe(
        data_list,
        start_idx,
        end_idx,
        sequence_proportions_1,
        sequence_proportions_2,
        sequence_proportions_3,
    )

    print("Combining aggregated dataframe and labels dataframe...")
    aggregated_df["transcript_position"] = aggregated_df["transcript_position"].astype(
        int
    )
    labels_df["transcript_position"] = labels_df["transcript_position"].astype(int)
    df_full = get_full_dataframe(aggregated_df, labels_df)
    X, y = get_X_Y(df_full)

    print("Performing train test split and scaling features...")
    X_train, X_test, y_train, y_test = train_test_split_by_geneid(X, y)
    X_train_scaled, X_test_scaled = scale_X_features(X_train, X_test)

    print("Getting best parameters...")
    best_params = {
        "subsample": 0.9,
        "n_estimators": 1100,
        "min_child_weight": 6,
        "max_depth": 13,
        "max_delta_step": 0,
        "learning_rate": 0.01,
        "gamma": 0,
        "colsample_bytree": 0.8,
    }
    pos_weights = get_scale_pos_weights(y["label"])

    print("Training model...")
    model = XGBClassifier(
        scale_pos_weight=pos_weights,
        eval_metric="logloss",
        **best_params,
    )
    model.fit(X_train_scaled, y_train)

    print("Saving model...")
    model_save_path = os.path.join(output_path, output_filename) + ".pkl"
    with open(model_save_path, "wb") as f:
        pickle.dump(model, f)

    print(f"Model saved to {model_save_path}")

    return model


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train an m6A prediction model")
    parser.add_argument("--data", required=True, help="Path to RNA-seq data JSON file")
    parser.add_argument("--labels", required=True, help="Path to m6A labels file")
    parser.add_argument(
        "--output_path", required=True, help="Path to save the trained model"
    )
    parser.add_argument(
        "--output_file", required=True, help="Filename to save the trained model"
    )
    args = parser.parse_args()

    train_model(args.data, args.labels, args.output_path, args.output_file)
