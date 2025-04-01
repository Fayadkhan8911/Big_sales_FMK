#%%
import pandas as pd
import numpy as np  
import matplotlib.pyplot as plt
import seaborn as sns
# %%
df=pd.read_csv('../data/Encoded_Data.csv')
df.head()
# %%
df.describe()
# %% convert the item_visibility to a categorical variable 
df['Item_Visibility_cat'] = pd.cut(df['Item_Visibility'], bins=5, labels=['Very Low', 'Low', 'Medium', 'High', 'Very High'])

# %%
df.head()
# %%
df.to_csv('../data/Cleaned Data.csv',index=False)

#%%
df['Item_Visibility_Class'].value_counts()
# %%
df.info()
# %%
import matplotlib.pyplot as plt
import numpy as np

# Get feature importance
feature_importance = model.feature_importances_

# Get feature names (ensure they match the trained model)
feature_names = model.get_booster().feature_names

# Convert to DataFrame for easy visualization
importance_df = pd.DataFrame({'Feature': feature_names, 'Importance': feature_importance})
importance_df = importance_df.sort_values(by='Importance', ascending=False)

# Display in Streamlit
st.subheader("Feature Importance")
st.dataframe(importance_df)

# Plot feature importance
fig, ax = plt.subplots()
ax.barh(importance_df['Feature'], importance_df['Importance'], color='skyblue')
ax.set_xlabel("Importance")
ax.set_ylabel("Feature")
ax.set_title("Feature Importance")
st.pyplot(fig)
