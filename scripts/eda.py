# Item_Type and Outlet_Type, Outlet_Size are imbalanced 
# Drop Item_Identifier  


#%%
import pandas as pd
import numpy as np  
import matplotlib.pyplot as plt
import seaborn as sns
# %%
df=pd.read_csv('/media/di-moriarty/Bankai/Work/Projects/Sales/data/Big Sales Data.csv')
df.head()
# %%
df.info()
# %%
df.describe()
# %%

df['Item_Weight'].fillna(12.5,inplace=True)

# %%
df.columns
# %%
# from ydata_profiling import ProfileReport
# profile = ProfileReport(df, title='Sales Report', explorative=True)
# profile.to_widgets()
# %%
### Unnecessary columns
df.drop(columns=['Item_Identifier','Outlet_Identifier'],inplace=True)
# %%

df['Item_Fat_Content'].value_counts()
# %%
df['Item_Fat_Content'].replace({'LF':'Low Fat','reg':'Regular','low fat':'Low Fat'},inplace=True)
# %%
df['Item_Fat_Content'].value_counts()
# %%
df['Item_Type'].value_counts()
# %%
df['Outlet_Size'].value_counts()
# %%
df['Outlet_Size'].fillna('Medium',inplace=True)
# %%
df['Outlet_Location_Type'].value_counts()
# %%
df['Outlet_Type'].value_counts()
# %%
df['Item_Type_Combined'] = df['Item_Identifier'].apply(lambda x: x[0:2])
df['Item_Type_Combined'] = df['Item_Type_Combined'].map({'FD':'Food','NC':'Non-Consumable','DR':'Drinks'})
df['Item_Type_Combined'].value_counts()
# %%
df.to_csv('/media/di-moriarty/Bankai/Work/Projects/Sales/data/Cleaned Data.csv',index=False)
# %%