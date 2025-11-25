import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
import os

curr_dir = os.path.dirname(os.path.abspath(__file__))

def return_preprocessed_lead_scoring_data(df):
    df["qualified_lead"] = (
        (df.get("deal_stage_ENGAGING", 0) == 1) |
        (df.get("deal_stage_WON", 0) == 1) |
        (df.get("deal_stage_LOST", 0) == 1)
    ).astype(int)
    TARGET = "qualified_lead"
    temporal_cols = ['engage_date', 'close_date', 'engage_year', 'engage_month', 
                    'engage_dayofweek', 'days_to_close', 'closed_within_30d']
    outcome_cols = ['deal_stage_PROSPECTING', 'deal_stage_ENGAGING', 
                    'deal_stage_WON', 'deal_stage_LOST', 'won_deal', 
                    'has_close_date', 'close_value', 'close_value_log']
    remove_cols = temporal_cols + outcome_cols
    raw_features = ['revenue', 'employees', 'sales_price']
    feature_cols = [c for c in df.columns if c not in remove_cols + raw_features + [TARGET]]
    X = df[feature_cols]
    return X.copy()

df = pd.read_csv(os.path.join(curr_dir, "..", "..", "data", "opp_data_with_id.csv"))
df_clean = pd.read_csv(os.path.join(curr_dir, "..", "..", "data", "cleaned_data_with_id.csv"))
print(df)
df_clean = df_clean[df_clean['opportunity_id'] == "PE84CX4O"]
X = return_preprocessed_lead_scoring_data(df_clean)
X.drop("opportunity_id", axis=1,inplace=True)
print(X)
import pickle
model = pickle.load(open(os.path.join(curr_dir, "..", "..", "models", "lead_scoring_model.pkl"), "rb"))['classifier']
print(model.predict(X))
