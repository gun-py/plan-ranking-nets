import argparse
import pandas as pd
import numpy as np
import torch
from torch.optim import LBFGS
import matplotlib.pyplot as plt
import dill
from utils import *
from model import *
from preprocess_knn_embeddings import *

def main(input_path, preds_name):
    model_dir = './model_weights'
    loss_dir = './plackett_luce'
    
    check18_cols = ['0_Deductible - Individual',
                    '0_Out-Of-Pocket Max - Individual',
                    '0_Coinsurance',
                    '0_HDHP?',
                    '0_Specialist_Visit_transformed',
                    '0_Urgent_care_transformed',
                    '0_Emergency_Room_transformed',
                    '0_Hospital_Stay_transformed',
                    '0_Doctor_Visit_price_after_ded_combined',
                    '1_Deductible - Individual',
                    '1_Out-Of-Pocket Max - Individual',
                    '1_Coinsurance',
                    '1_HDHP?',
                    '1_Specialist_Visit_transformed',
                    '1_Urgent_care_transformed',
                    '1_Emergency_Room_transformed',
                    '1_Hospital_Stay_transformed',
                    '1_Doctor_Visit_price_after_ded_combined']

    df_full = pd.read_csv(input_path)

    required_columns = ['0_ID', '1_ID']
    if not all(col in df_full.columns for col in required_columns):
        raise ValueError(f"Columns {required_columns} are missing in the input CSV.")

    df = df_full[check18_cols]

    try:
        assert list(df.columns) == check18_cols, f"Expected columns: {check18_cols}, but got: {list(df.columns)}"
        print("The columns match the expected structure.")
    except AssertionError as e:
        print(f"AssertionError: {e}")
        raise

    test = df.values

    models = load_models(model_dir)
    plackett_luce = load_models(loss_dir)

    preds_a, preds_b = base_model(models, test)
    df_full['labels0'] = preds_a
    df_full['labels1'] = preds_b

    check_data = df_full[['0_ID', '1_ID', 'labels0', 'labels1']].copy()
    cured_data = []

    for _, row in check_data.iterrows():
        diff = row['labels1'] - row['labels0']
        if diff > 0:
            cured_data.append([row['1_ID'], row['0_ID'], diff])
        else:
            cured_data.append([row['0_ID'], row['1_ID'], abs(diff)])

    run_scores_def = load_function('get_plan_scores_def.pkl')
    plan_scores = run_scores_def(cured_data, plackett_luce, test)

    save_dict_to_json(plan_scores, preds_name)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Process input path and prediction name.')
    parser.add_argument('--input_path', type=str, required=True, help='Path to input CSV file')
    parser.add_argument('--preds_name', type=str, required=True, help='Name for the predictions JSON file')

    args = parser.parse_args()
    main(args.input_path, args.preds_name)
