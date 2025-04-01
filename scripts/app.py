import streamlit as st
import joblib
import pandas as pd
from sklearn.preprocessing import LabelEncoder

# Load the trained model
model = joblib.load('xgb_model.joblib')  # Adjust the path as needed

# Streamlit app header
st.title('Item Outlet Sales Prediction')
st.write('This app predicts the outlet sales based on the provided features.')
st.write('Here XGBoost model was used.')


# Define the inputs for the app
item_weight = st.number_input('Item Weight', min_value=4.5, value=30.0)
item_mrp = st.number_input('Item MRP', min_value=31.30, value=270.0)

#item visibility

vis_options=['Very Low', 'Low', 'Medium', 'High', 'Very High']
item_visibility = st.selectbox('Item visibility', options=vis_options)

# Item Type options
item_type_options = ['Baking Goods', 'Breads', 'Breakfast', 'Canned', 'Dairy', 'Frozen Foods', 
                     'Fruits and Vegetables', 'Hard Drinks', 'Health and Hygiene', 'Household', 
                     'Meat', 'Others', 'Seafood', 'Snack Foods', 'Soft Drinks', 'Starchy Foods']
item_type = st.selectbox('Item Type', options=item_type_options)

# Outlet Size options
outlet_size_options = ['Medium', 'Small', 'High']
outlet_size = st.selectbox('Outlet Size', options=outlet_size_options)

# Outlet Location Type options
outlet_location_type_options = ['Tier 1', 'Tier 2', 'Tier 3']
outlet_location_type = st.selectbox('Outlet Location Type', options=outlet_location_type_options)

# Outlet Type options
outlet_type_options = ['Supermarket Type1', 'Supermarket Type2', 'Grocery Store', 'Supermarket Type3']
outlet_type = st.selectbox('Outlet Type', options=outlet_type_options)

# Initialize label encoders for categorical variables
label_encoder_item_type = LabelEncoder()
label_encoder_outlet_size = LabelEncoder()
label_encoder_outlet_location_type = LabelEncoder()
label_encoder_outlet_type = LabelEncoder()
label_encoder_vis_type=LabelEncoder ()

# Fit the encoders on the categories (This should be done during training ideally)
label_encoder_item_type.fit(item_type_options)
label_encoder_outlet_size.fit(outlet_size_options)
label_encoder_outlet_location_type.fit(outlet_location_type_options)
label_encoder_outlet_type.fit(outlet_type_options)
label_encoder_vis_type.fit(vis_options)

# Encode the categorical inputs
item_type_encoded = label_encoder_item_type.transform([item_type])[0]
outlet_size_encoded = label_encoder_outlet_size.transform([outlet_size])[0]
outlet_location_type_encoded = label_encoder_outlet_location_type.transform([outlet_location_type])[0]
outlet_type_encoded = label_encoder_outlet_type.transform([outlet_type])[0]
item_visibility=label_encoder_vis_type.transform([item_visibility])[0]

# Create a DataFrame for the inputs to match the model's input format
input_data = pd.DataFrame({
    'Item_Weight': [item_weight],
    'Item_MRP': [item_mrp],
    'Item_Visibility': [item_visibility],
    'Item_Type': [item_type_encoded],
    'Outlet_Size': [outlet_size_encoded],
    'Outlet_Location_Type': [outlet_location_type_encoded],
    'Outlet_Type': [outlet_type_encoded]
})

# Make prediction
if st.button('Predict'):
    # Predict the sales based on the inputs
    prediction = model.predict(input_data)
    
    # Show the predicted sales
    st.write(f'Predicted Item Outlet Sales: ${prediction[0]:.2f}')
