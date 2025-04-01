#%%
import pandas as pd
import numpy as np  
import matplotlib.pyplot as plt
import seaborn as sns
# %%
df=pd.read_csv('/media/di-moriarty/Bankai/Work/Projects/Sales/data/Encoded_Data.csv')
# df.head()
df.columns
#%%
df.describe()

# %%
from ydata_profiling import ProfileReport
profile = ProfileReport(df, title='Sales Report', explorative=True)
# profile.to_file('/media/di-moriarty/Bankai/Work/Projects/Sales/data/Sales Report.html')
#%%
df.head()
# %%
df['Item_Visibility'].replace(0, df['Item_Visibility'].median(), inplace=True)
#%%
bins = np.linspace(0.003574698, 0.328390948, 6)
labels = ['Very Low', 'Low', 'Medium', 'High', 'Very High']

# Create a new categorical column based on the bins
df['Item_Visibility_Class'] = pd.cut(df['Item_Visibility'], bins=bins, labels=labels)

df.head()
#%%
from sklearn.preprocessing import LabelEncoder

# Initialize LabelEncoder
label_encoder = LabelEncoder()

# # Encode the 'Item_Visibility_Class' column
# df['Item_Visibility_Class'] = label_encoder.fit_transform(df['Item_Visibility_Class'])

# # Drop the 'Item_Visibility' column
# df = df.drop(columns=['Item_Visibility'])

df.head()

#%% 
df.info()

#%%
# df.to_csv('/media/di-moriarty/Bankai/Work/Projects/Sales/data/Encoded_Data.csv',index=False)
df['Item_Visibility_Class'].value_counts()
#%% train test split
columns=['Item_Weight', 'Item_MRP', 'Item_Visibility', 'Item_Type','Outlet_Size', 'Outlet_Location_Type', 'Outlet_Type']
x=df[columns]
x.head()
# %%
for col in x.select_dtypes(include=['object']).columns:
    x[col] = x[col].astype('category')
#%%
# from ydata_profiling import ProfileReport
# profile = ProfileReport(x, title='Selected Report', explorative=True)
# profile.to_file('/media/di-moriarty/Bankai/Work/Projects/Sales/data/Selected Report.html')

# %%
from sklearn.preprocessing import LabelEncoder
label_encoders = {}
for col in x.select_dtypes(include=['category']).columns:
    le = LabelEncoder()
    x[col] = le.fit_transform(x[col])  # Convert categories to numbers
    label_encoders[col] = le 


# %%
# x.to_csv('/media/di-moriarty/Bankai/Work/Projects/Sales/data/Encoded_Data.csv',index=False)
x['Item_Type'].value_counts()   
# %%
