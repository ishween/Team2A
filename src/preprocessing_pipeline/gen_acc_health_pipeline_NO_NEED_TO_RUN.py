# import pandas as pd
# import numpy as np
# from sklearn.preprocessing import StandardScaler, LabelEncoder, MinMaxScaler
# from sklearn.impute import SimpleImputer
# import os

# def return_preprocessed_acc_health(df: pd.DataFrame) -> pd.DataFrame:
  
#     ids = df['opportunity_id'] if 'opportunity_id' in df.columns else None
#     np.random.seed(42)

#     def robust_minmax(s):
#         s = pd.to_numeric(s, errors='coerce')
#         ql, qh = s.quantile(0.02), s.quantile(0.98)
#         s = s.clip(ql, qh)
#         mm = MinMaxScaler()
#         return pd.Series(mm.fit_transform(s.values.reshape(-1,1)).ravel(), index=s.index)

#     # Revenue
#     rev = pd.to_numeric(df.get("revenue"), errors="coerce")
#     rev_01 = robust_minmax(rev.fillna(rev.median()))

#     # Revenue per employee
#     rpe = pd.to_numeric(df.get("rev_per_employee"), errors="coerce")
#     rpe_01 = robust_minmax(rpe.fillna(rpe.median()))

#     # Employees
#     emp = pd.to_numeric(df.get("employees"), errors="coerce")
#     emp_01 = robust_minmax(emp.fillna(emp.median()))

#     # Recency index for engagement activity
#     # calculation: 1 / (1 + days_since_last_close), where days_since_last_close = today - close_date; if close_date is missing, the max value (worst recency) is used
#     if "close_date" in df.columns:
#         close_dt = pd.to_datetime(df["close_date"], errors="coerce")
#         today = pd.Timestamp.today().normalize()
#         days = (today - close_dt).dt.days
#         # fill missing with max (worst recency)
#         max_days = int(days.max()) if np.isfinite(days.max()) else 30
#         days = days.fillna(max_days).clip(lower=0)

#     recency_index = 1.0 / (1.0 + days)
#     recency_01 = robust_minmax(recency_index)

#     # Deal margin:
#     # calculation: (close_value - sales_price) / sales_price
#     close_value = pd.to_numeric(df.get("close_value"), errors="coerce")
#     sales_price = pd.to_numeric(df.get("sales_price"), errors="coerce")
#     margin = (close_value - sales_price) / sales_price.replace(0, np.nan)
#     margin = margin.replace([np.inf, -np.inf], np.nan).fillna(0)
#     margin_01 = robust_minmax(margin)

#     # Winrate adj
#     # calculation: account_win_rate * log(1 + agent_closed_deals)
#     acc_win_rate = pd.to_numeric(df.get("account_win_rate"), errors="coerce").fillna(0)
#     agent_closed = pd.to_numeric(df.get("agent_closed_deals"), errors="coerce").fillna(0)
#     winrate_adj = acc_win_rate * np.log1p(agent_closed)
#     winrate_adj_01 = robust_minmax(winrate_adj)

#     # Product variety
#     # calculation: the sum of listed product columns (how many DISTINCT lines are purchased)
#     product_cols = [
#         "product_GTK 500",
#         "product_GTX BASIC",
#         "product_GTX PLUS BASIC",
#         "product_GTX PLUS PRO",
#         "product_GTX PRO",
#         "product_MG ADVANCED",
#         "product_MG SPECIAL",
#     ]
#     # handling any missing product_* columns
#     available_products = [c for c in product_cols if c in df.columns]
#     if available_products:
#         # count DISTINCT lines that have been purchased (>0)
#         prod_variety = (df[available_products].fillna(0) > 0).sum(axis=1)
#     else:
#         prod_variety = pd.Series(0, index=df.index)
#     prod_variety_01 = robust_minmax(prod_variety)

#     # assigning weights, which sum to 1
#     w = {
#         'revenue':          0.18,
#         'rev_per_employee': 0.18,
#         'employees':        0.08,
#         'recency':          0.18,
#         'margin':           0.13,
#         'winrate_adj':      0.15,
#         'product_variety':  0.10
#     }

#     # final account health score computation
#     # adding some noise so the model has something to learn
#     noise = np.random.normal(0, 0.02, size=len(df))
#     score_raw = (
#         w['revenue']          * rev_01.values +
#         w['rev_per_employee'] * rpe_01.values +
#         w['employees']        * emp_01.values +
#         w['recency']          * recency_01.values +
#         w['margin']           * margin_01.values +
#         w['winrate_adj']      * winrate_adj_01.values +
#         w['product_variety']  * prod_variety_01.values +
#         noise
#     )

#     # scaling to 0–100
#     score_01 = robust_minmax(pd.Series(score_raw, index=df.index))
#     df['account_health_score'] = (score_01 * 100).round(2)
#     df['account_health_score'] = pd.to_numeric(df['account_health_score'], errors='coerce') \
#                                     .replace([np.inf, -np.inf], np.nan) \
#                                     .fillna(df['account_health_score'].median())

#     # keep working dataframe named df for the next steps
#     print("Synthetic target created. Summary:")

#     #deciding on features for the model

#     target_col = 'account_health_score'
#     #numeric features
#     X_full = df.select_dtypes(include=[np.number]).copy()
#     #dropping IDs, dates (unique identifiers)
#     drop_like = ['id','uuid','account_id','customer_id','date','created','updated','timestamp']
#     to_drop = set([c for c in X_full.columns for k in drop_like if k in c.lower()])
#     to_drop.add(target_col)

#     feature_pool = [c for c in X_full.columns if c not in to_drop]
#     X_pool = X_full[feature_pool].replace([np.inf,-np.inf], np.nan)
#     X_pool = X_pool.fillna(X_pool.median(numeric_only=True))
#     #checking for columns with >0.95 correlation and dropping them
#     corr = X_pool.corr().abs()
#     upper = corr.where(np.triu(np.ones(corr.shape), k=1).astype(bool))
#     high_corr_drop = [column for column in upper.columns if any(upper[column] > 0.95)]
#     X_dedup = X_pool.drop(columns=high_corr_drop, errors='ignore')


#     y = pd.to_numeric(df[target_col], errors='coerce')
#     mask = y.notna() & np.isfinite(y)
#     X = X_dedup.loc[mask].copy()
#     y = y.loc[mask].copy()
#     #using the top top_k features with the highest correlation to target to train the model
#     #we do this because features with the highest correlation to target are the variables that vary most strongly with the target;
#     #they are the features with the greatest influence on the target
#     corr_to_y = X.apply(lambda s: s.corr(y)).abs().sort_values(ascending=False)
#     top_k = min(50, len(corr_to_y))  #can change top_k if we want to
#     selected_features = list(corr_to_y.head(top_k).index)

#     X = X[selected_features].copy()

#     scaler_X = StandardScaler()
#     X_scaled = scaler_X.fit_transform(X)
#     X_preprocessed = pd.DataFrame(X_scaled, columns=X.columns, index=X.index)
#     X_preprocessed['opportunity_id'] = ids
#     X_preprocessed.to_csv(os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..", "data", "preprocessed_acc_health.csv"), index=False)

# curr_dir = os.path.dirname(os.path.abspath(__file__))

# # df = pd.read_csv(os.path.join(curr_dir, "..", "..", "data", "opp_data_with_id.csv"))
# # df_clean = pd.read_csv(os.path.join(curr_dir, "..", "..", "data", "preprocessed_acc_health.csv"))
# # print(df)
# # X = df_clean[df_clean['opportunity_id'] == "PE84CX4O"]
# # # return_preprocessed_acc_health(df_clean)

# # X.drop("opportunity_id", axis=1,inplace=True)
# # print(X)
# # import pickle
# # model = pickle.load(open(os.path.join(curr_dir, "..", "..", "models", "account_health_model.pkl"), "rb"))['model']
# # print(model.predict(X))


