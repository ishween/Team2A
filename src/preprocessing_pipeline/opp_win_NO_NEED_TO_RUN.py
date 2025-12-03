# import pandas as pd
# import numpy as np
# from sklearn.preprocessing import StandardScaler
# from sklearn.impute import SimpleImputer
# import os

# curr_dir = os.path.dirname(os.path.abspath(__file__))

# def return_preprocessed_opp_win_data(df):
# # df = pd.read_csv("../data/processed/cleaned_data.csv")
#     leakage_features = [
#         'close_date',         
#         'close_value',      
#         'has_close_date',   
#         'close_value_log',   
#         'days_to_close',       
#         'closed_within_30d',    
#         'deal_stage_WON',       
#         'deal_stage_LOST',      
#         'deal_stage_ENGAGING',  
#         'deal_stage_PROSPECTING',
        
#         'agent_closed_deals',   
#         'account_win_rate',     
        
#         'deal_stage',
#         'account',              
#         'sales_agent',          
#     ]

#     leakage_removed = [col for col in leakage_features if col in df.columns]
#     df_clean = df.drop(columns=leakage_removed, errors='ignore')

#     if 'engage_date' in df_clean.columns:
#         df_clean['engage_date'] = pd.to_datetime(df_clean['engage_date'], errors='coerce')
#         df_clean['engage_year'] = df_clean['engage_date'].dt.year
#         df_clean['engage_month'] = df_clean['engage_date'].dt.month
#         df_clean['engage_day_of_week'] = df_clean['engage_date'].dt.dayofweek
#         df_clean['engage_quarter'] = df_clean['engage_date'].dt.quarter
#         df_clean['is_month_end'] = (df_clean['engage_date'].dt.day > 25).astype(int)
#         df_clean['is_quarter_end'] = (df_clean['engage_month'] % 3 == 0).astype(int)
        

#     if 'year_established' in df_clean.columns:
#         df_clean['company_age'] = 2017 - df_clean['year_established']
#         print("✅ Created company_age feature")

#     if 'revenue' in df_clean.columns and 'employees' in df_clean.columns:
#         df_clean['revenue_per_employee'] = df_clean['revenue'] / (df_clean['employees'] + 1)

#     if 'sales_price' in df_clean.columns and 'revenue' in df_clean.columns:
#         df_clean['price_to_revenue_ratio'] = df_clean['sales_price'] / (df_clean['revenue'] + 1)

#     if 'revenue' in df_clean.columns:
#         df_clean['revenue_log'] = np.log1p(df_clean['revenue'])
        
#     if 'employees' in df_clean.columns:
#         df_clean['employees_log'] = np.log1p(df_clean['employees'])
        
#     if 'sales_price' in df_clean.columns:
#         df_clean['sales_price_log'] = np.log1p(df_clean['sales_price'])
        
#     target = 'won_deal'
#     y = df_clean[target].astype(int)

#     cols_to_drop = [
#         target,
#         'engage_date', 
#         'office_location', 
#     ]

#     ids = df_clean['opportunity_id'] if 'opportunity_id' in df_clean.columns else None
#     X = df_clean.drop(columns=[col for col in cols_to_drop if col in df_clean.columns])
#     numeric_features = X.select_dtypes(include=['number']).columns.tolist()
#     X = X[numeric_features]

#     imputer = SimpleImputer(strategy='median')
#     X_imputed = imputer.fit_transform(X)
#     X = pd.DataFrame(X_imputed, columns=X.columns, index=X.index)

#     # Scale features - fit only on training data!
#     scaler = StandardScaler()
#     X_scaled = scaler.fit_transform(X)

#     final = pd.DataFrame(X_scaled, columns=X.columns, index=X.index)
#     final['opportunity_id'] = ids
#     final.to_csv(os.path.join(curr_dir, "..", "..", "data", "preprocessed_opp_win_data.csv"), index=False)
#     return final


# # df = pd.read_csv(os.path.join(curr_dir, "..", "..", "data", "opp_data_with_id.csv"))
# df_clean = pd.read_csv(os.path.join(curr_dir, "..", "..", "data", "preprocessed_opp_win_data.csv"))
# X = df_clean[df_clean['opportunity_id'] == "PE84CX4O"].copy()
# X.drop('opportunity_id', axis=1, inplace=True)

# import pickle
# model = pickle.load(open(os.path.join(curr_dir, "..", "..", "models", "opportunity_win_model.pkl"), "rb"))
# print(model.predict(X))


