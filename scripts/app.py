import streamlit as st
import joblib
import pandas as pd
import xgboost as xgb
from sklearn.preprocessing import LabelEncoder

# Load the trained model
model = joblib.load('xgb_model.joblib')  # Adjust the path as needed

# Streamlit app header
st.title('Item Outlet Sales Prediction')
st.write('This app predicts the outlet sales based on the provided features using an XGBoost model.')

# Define user input fields
item_weight = st.number_input('Item Weight', min_value=4.5, value=30.0)
item_mrp = st.number_input('Item MRP', min_value=31.30, value=270.0)

# Item Visibility (Categorical Encoding)
vis_options = ['Very Low', 'Low', 'Medium', 'High', 'Very High', 'Extra High']
item_visibility = st.selectbox('Item Visibility', options=vis_options)

# Item Type
item_type_options = ['Baking Goods', 'Breads', 'Breakfast', 'Canned', 'Dairy', 'Frozen Foods', 
                     'Fruits and Vegetables', 'Hard Drinks', 'Health and Hygiene', 'Household', 
                     'Meat', 'Others', 'Seafood', 'Snack Foods', 'Soft Drinks', 'Starchy Foods']
item_type = st.selectbox('Item Type', options=item_type_options)

# Outlet Size
outlet_size_options = ['Medium', 'Small', 'High']
outlet_size = st.selectbox('Outlet Size', options=outlet_size_options)

# Outlet Location Type
outlet_location_type_options = ['Tier 1', 'Tier 2', 'Tier 3']
outlet_location_type = st.selectbox('Outlet Location Type', options=outlet_location_type_options)

# Outlet Type
outlet_type_options = ['Supermarket Type1', 'Supermarket Type2', 'Grocery Store', 'Supermarket Type3']
outlet_type = st.selectbox('Outlet Type', options=outlet_type_options)

# Initialize Label Encoders
label_encoder_item_type = LabelEncoder()
label_encoder_outlet_size = LabelEncoder()
label_encoder_outlet_location_type = LabelEncoder()
label_encoder_outlet_type = LabelEncoder()
label_encoder_vis_type = LabelEncoder()

# Fit label encoders (Ensure consistency with training)
label_encoder_item_type.fit(item_type_options)
label_encoder_outlet_size.fit(outlet_size_options)
label_encoder_outlet_location_type.fit(outlet_location_type_options)
label_encoder_outlet_type.fit(outlet_type_options)
label_encoder_vis_type.fit(vis_options)

# Encode categorical inputs
item_type_encoded = label_encoder_item_type.transform([item_type])[0]
outlet_size_encoded = label_encoder_outlet_size.transform([outlet_size])[0]
outlet_location_type_encoded = label_encoder_outlet_location_type.transform([outlet_location_type])[0]
outlet_type_encoded = label_encoder_outlet_type.transform([outlet_type])[0]
item_visibility_encoded = label_encoder_vis_type.transform([item_visibility])[0]

# Create DataFrame for input
input_data = pd.DataFrame({
    'Item_Weight': [item_weight],
    'Item_MRP': [item_mrp],
    'Item_Visibility': [item_visibility_encoded],  # Ensure correct feature name
    'Item_Type': [item_type_encoded],
    'Outlet_Size': [outlet_size_encoded],
    'Outlet_Location_Type': [outlet_location_type_encoded],
    'Outlet_Type': [outlet_type_encoded]
})

# Ensure feature order matches the trained model
expected_features = model.get_booster().feature_names
if 'Item_Visibility_Class' in expected_features:
    input_data.rename(columns={'Item_Visibility': 'Item_Visibility_Class'}, inplace=True)

# Convert data to float (XGBoost requires numeric input)
input_data = input_data.astype(float)

# # Debugging: Display expected vs actual feature names
# st.subheader("Feature Alignment Check")
# st.write("Expected features by model:", expected_features)
# st.write("Provided features:", list(input_data.columns))

# Make Prediction
# Ensure input data follows the correct feature order
expected_order = ['Item_Weight', 'Item_MRP', 'Item_Type', 'Outlet_Size', 'Outlet_Location_Type', 'Outlet_Type', 'Item_Visibility_Class']
input_data = input_data[expected_order]  # Fix feature order mismatch

# Make Prediction
if st.button('Predict'):
    try:
        # Predict sales
        prediction = model.predict(input_data)

        # Show result
        st.success(f'Predicted Item Outlet Sales: ${prediction[0]:.2f}')
    
    except ValueError as e:
        st.error(f"Feature Mismatch Error: {e}")

