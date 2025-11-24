import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

df = pd.read_csv("../data/processed/cleaned_data.csv")

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
y = df[TARGET]

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

# TODO create this function, return PREPROCESSED X for model training