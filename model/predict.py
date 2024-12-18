import joblib
import argparse
import os

from sklearn.preprocessing import StandardScaler
from utils import (
    read_data_json,
    get_5_mer_sequence_proportions,
    create_aggregated_dataframe,
)


def predict_m6a(data_json_path, model_path, output_path, save_info):
    """
    Applies a pre-trained model to predict m6A modification sites on new RNA-Seq data and saves prediction results to specified path.

    Args:
        data_json_path (str): Path to data json file.
        model_path (str): Path to the pre-trained model file.
        output_path (str): Path to save the prediction results.

    Returns:
        output_df (pd.DataFrame): Dataframe contaning prediction results. 
    """

    with open(model_path, "rb") as f:
        model = joblib.load(f)

    print("Reading data json...")
    data_list = read_data_json(data_json_path)

    print("Computing sequence proportions...")
    sequence_proportions_1, sequence_proportions_2, sequence_proportions_3 = (
        get_5_mer_sequence_proportions(data_list)
    )

    print("Performing feature engineering...")
    start_idx = 0
    end_idx = 5
    X_test_df = create_aggregated_dataframe(
        data_list,
        start_idx,
        end_idx,
        sequence_proportions_1,
        sequence_proportions_2,
        sequence_proportions_3,
    )

    if save_info:
        output_df = X_test_df[['transcript_id', 'transcript_position', 'sequence', 
                                'proportion_1','proportion_2', 'proportion_3', 
                            'diff_1_1', 'diff_1_2', 'diff_1_3', 
                            'diff_2_1', 'diff_2_2', 'diff_2_3', 
                            'length', 
                            'mean_0','var_0', 'max_0', 'min_0', 
                            'mean_1', 'var_1', 'max_1', 'min_1', 
                            'mean_2', 'var_2', 'max_2', 'min_2', 
                            'mean_3', 'var_3', 'max_3', 'min_3', 
                            'mean_4', 'var_4', 'max_4', 'min_4', 
                            'mean_5', 'var_5', 'max_5', 'min_5', 
                            'mean_6', 'var_6', 'max_6', 'min_6', 
                            'mean_7', 'var_7', 'max_7', 'min_7', 
                            'mean_8', 'var_8', 'max_8', 'min_8', ]]
    else:
        output_df = X_test_df[["transcript_id", "transcript_position"]]

    print("Scaling features...")
    X_test_df = X_test_df.drop(["transcript_id", "transcript_position", "sequence"], axis=1)
    scaler = StandardScaler()
    X_test_scaled = scaler.fit_transform(X_test_df)

    print("Predicting m6a...")
    y_pred_proba = model.predict_proba(X_test_scaled)[:, 1]
    output_df["score"] = y_pred_proba

    save_df_path = output_path
    output_df.to_csv(save_df_path, index=False)

    print(f"Prediction results saved to {save_df_path}")

    return output_df


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Predict m6A modification sites")
    parser.add_argument("--data", required=True, help="Path to RNA-seq data JSON file")
    parser.add_argument("--model", required=True, help="Path to the trained model file")
    parser.add_argument("--output_path", required=True, help="Path to save predictions")
    parser.add_argument("--save_info", required=False, help="Save additional information such as sequence and features")
    args = parser.parse_args()

    predict_m6a(args.data, args.model, args.output_path, args.save_info)

# To run with save_info: (for Task 2, data analysis)
# python model/predict.py --data path/to/data.json --model path/to/model.pkl --output_path path/to/output.csv --save_info True
