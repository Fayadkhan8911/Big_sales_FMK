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
import numpy as np
import xgboost as xgb
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import mean_squared_error

# Define XGBoost model
model = xgb.XGBRegressor(enable_categorical=True, eval_metric="rmse")

#%%
param_grid = {
    "n_estimators": [200, 300, 500],      # Number of boosting rounds
    "learning_rate": [0.01, 0.03, 0.05],  # Step size shrinkage
    "max_depth": [4, 6, 8],               # Tree depth
    "subsample": [0.8, 1.0],              # Row sampling
    "colsample_bytree": [0.8, 1.0],       # Feature sampling
    "alpha": [0.1, 0.5, 1.0],             # L1 regularization
    "lambda": [1.0, 2.0, 5.0]             # L2 regularization
}

# %%
grid_search = GridSearchCV(
    estimator=model,
    param_grid=param_grid,
    scoring="neg_root_mean_squared_error",  # RMSE
    cv=5,  # 5-fold cross-validation
    n_jobs=-1,
    verbose=1
)

grid_search.fit(x_train, y_train)

# %%
print(f"✅ Best Parameters: {grid_search.best_params_}")

best_model = grid_search.best_estimator_

# Predict on test data
y_test_pred = best_model.predict(x_test)

# Calculate RMSE & R²
final_test_rmse = np.sqrt(mean_squared_error(y_test, y_test_pred))
final_test_r2 = best_model.score(x_test, y_test)

print(f"📊 Final Test RMSE: {final_test_rmse:.4f}")
print(f"📊 Final Test R² Score: {final_test_r2:.4f}")

# %%
