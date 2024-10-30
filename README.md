# DSA4262 - Prediction of m6A RNA modifications from direct RNA-Seq data

This project aims to identify m6A RNA modifications from direct RNA sequencing data by leveraging on XGBoost and feature engineering.

The project structure is as follows.

```bash
main/
├── clustering/                    # Pickle files for clustering
├── notebooks/                     # Notebooks for testing
├── data/
│   ├── sample_rna_data.json       # Sample data for testing the prediction script
├── outputs/
│   ├── intermediate_leaderboard/  # Outputs for intermediate leaderboard
│   ├── model_predictions/         # Csv files containing model prediction outputs
├── .gitignore
├── README.md                      # Project documentation
├── model.pkl                      # Pickle file for pretrained model
├── predict.py                     # Script for making predictions
├── requirements.txt               # Project dependencies
├── train.py                       # Script for training the model
└── utils.py                       # Script containing functions used for training and prediction
```


# Table of Contents
- **[Installation](#installation)**<br>
- **[Running the Model](#running-the-model)**<br>
  - **[Model Training](#model-training)**<br>
  - **[Model Prediction](#model-prediction)**<br>
- **[Testing the Prediction Script](#testing-the-prediction-script)**<br>

# Installation
It is recommended to use [Python version 3.10 or higher](https://www.python.org). 

1. **Clone the Repository:**
   ```bash
   git clone https://github.com/yourusername/m6A-modification-prediction.git
   cd m6A-modification-prediction

2. **Create Virtual Environment:** (Optional but Recommended)
   ```bash
   python -m venv env
   source env/bin/activate
   
3. **Install Required Packages**
   ```bash
   pip install -r requirements.txt

# Running the Model

## Model Training

To train the model on RNA-Seq data:
```bash
python model/train.py --data path/to/rna_data.json --labels path/to/labels_data.info --output_path path/to/save_model.pkl
```
**Arguments:**
* ``--data``: Path to RNA-seq data, processed by m6Anet, in JSON format.
* ``--labels``: Path to the m6A labels file.
* ``--output_path``: Path to save the trained model file.

The trained model will be saved to the specified path in .pkl format for later use in predictions.

## Model Prediction

To run prediction on new data using the trained model:
```bash
python model/predict.py --data path/to/new_rna_data.json --model path/to/save_model.pkl --output_path path/to/predictions.csv
```
**Arguments:**
* ``--data``: Path to RNA-seq data, processed by m6Anet, in JSON format.
* ``--model``: Path to the trained model file.
* ``--output_path``: Path to save the prediction results to csv.

# Testing the Prediction Script

A sample dataset is included in data/sample_rna_data.json for testing the prediction script.
A pre-trained model is also included in model.pkl.

To run prediction on the sample json file:
```bash
python model/predict.py --data data/sample_rna_data.json --model model.pkl --output_path outputs/model_predictions/sample_predictions.json
```

