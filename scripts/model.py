#%%
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

#%%
x=pd.read_csv('/media/di-moriarty/Bankai/Work/Projects/Sales/data/Encoded_Data.csv',index_col=False)
df=pd.read_csv('/media/di-moriarty/Bankai/Work/Projects/Sales/data/Cleaned Data.csv',index_col=False)
y=df['Item_Outlet_Sales']
#%%


#%%
from sklearn.model_selection import train_test_split
x_train, x_test, y_train, y_test = train_test_split(x,y, test_size=0.2, random_state=42)


# %%



import xgboost as xgb
from xgboost import XGBRegressor
import pandas as pd

from sklearn.metrics import mean_squared_error, r2_score

#%% Train XGBoost with enable_categorical=True
# model = XGBRegressor(
#     enable_categorical=True,
#     n_estimators=300, 
#     learning_rate=0.05,
#     max_depth=4, 
#     eval_metric="rmse",
#     subsample=0.8,
#     alpha=0.1,
#     lambda_=1.0
#     ,early_stopping_rounds=50)
# {'alpha': 0.1, 'colsample_bytree': 0.8, 'lambda': 5.0, 'learning_rate': 0.01, 'max_depth': 4, 'n_estimators': 500, 'subsample': 0.8

model = XGBRegressor(
    enable_categorical=True,
    n_estimators=500,
    learning_rate=0.01,  # Lower learning rate for better generalization
    max_depth=6,  # Corrected from "max_dept"
    eval_metric="rmse",
    subsample=0.8,
    colsample_bytree=0.8,
    alpha=0.1,  # Reduced L1 regularization
    lambda_=5.0,  # Increased L2 regularization to prevent overfitting
    early_stopping_rounds=50
)




#%% Train with evaluation
eval_set = [(x_train, y_train), (x_test, y_test)]
model.fit(x_train, y_train, eval_set=eval_set, verbose=False)

#%% Extract RMSE values for each iteration
eval_results = model.evals_result()  # Get the evaluation results from the model
train_rmse = eval_results['validation_0']['rmse']
test_rmse = eval_results['validation_1']['rmse']

#%% Make final predictions
y_train_pred = model.predict(x_train)
y_test_pred = model.predict(x_test)


#%% Calculate final RMSE and R² scores
final_train_rmse = np.sqrt(mean_squared_error(y_train, y_train_pred))
final_test_rmse = np.sqrt(mean_squared_error(y_test, y_test_pred))
final_train_r2 = r2_score(y_train, y_train_pred)
final_test_r2 = r2_score(y_test, y_test_pred)

#%% Print Metrics
print(f"📊 Final Training RMSE: {final_train_rmse:.4f}")
print(f"📊 Final Test RMSE: {final_test_rmse:.4f}")
print(f"📊 Final Training R² Score: {final_train_r2:.4f}")
print(f"📊 Final Test R² Score: {final_test_r2:.4f}")

#%% Plot RMSE vs. Boosting Rounds
plt.figure(figsize=(10, 6))
plt.plot(train_rmse, label="Train RMSE", color='blue')
plt.plot(test_rmse, label="Test RMSE", color='red')
plt.xlabel("Boosting Rounds")
plt.ylabel("RMSE")
plt.title("XGBoost Training vs Test RMSE")
plt.legend()
plt.grid()
plt.show()

# %%Overfitting Check
if final_train_rmse < final_test_rmse and (final_train_rmse / final_test_rmse) < 0.8:
    print("\n⚠️ Overfitting Detected: The test error is significantly higher than the training error.")
elif final_train_rmse < final_test_rmse and (final_train_rmse / final_test_rmse) >= 0.8:
    print("\n✅ No Significant Overfitting: The model generalizes well.")
else:
    print("\n🚀 Good Fit: The test error is close to the training error.")
#%% saving the model
import joblib

# # Save the model using joblib
joblib.dump(model, 'xgb_model.joblib')
# %%
