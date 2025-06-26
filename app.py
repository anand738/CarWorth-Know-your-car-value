import streamlit as st
import pandas as pd
from joblib import load
from sklearn.preprocessing import LabelEncoder
from sklearn.compose import ColumnTransformer

# Cache model and encoder loading to improve performance
@st.cache_resource
def load_resources():
    try:
        model = load('model.joblib')
        preprocessor = load('preprocessor.joblib')
        le_brand = load('label_encoder_brand.joblib')
        le_model = load('label_encoder_model.joblib')
        return model, preprocessor, le_brand, le_model
    except FileNotFoundError as e:
        st.error(f"Error loading model or encoders: {str(e)}")
        st.stop()

# Load resources
model, preprocessor, le_brand, le_model = load_resources()

# Brand and model mappings (corrected for spaces, casing, and duplicates)
brand_model_mapping = {
    'Maruti': ['Alto', 'Wagon R', 'Swift', 'Ciaz', 'Baleno', 'Swift Dzire', 'Ignis', 'Vitara', 'Celerio', 'Ertiga', 'Eeco', 'Dzire Vxi', 'Xl6', 'S-Presso', 'Dzire Lxi', 'Dzire Zxi'],
    'Hyundai': ['Grand', 'I20', 'I10', 'Venue', 'Verna', 'Creta', 'Santro', 'Elantra', 'Aura', 'Tucson'],
    'Honda': ['City', 'Amaze', 'Civic', 'CR-V', 'Jazz'],
    'Toyota': ['Innova', 'Fortuner', 'Camry', 'Corolla', 'Yaris'],
    'Ford': ['EcoSport', 'Endeavour', 'Figo', 'Aspire', 'Mustang'],
    'Volkswagen': ['Polo', 'Vento', 'Tiguan', 'Passat', 'Ameo'],
    'Tata': ['Nexon', 'Harrier', 'Safari', 'Tiago', 'Altroz'],
    'Mahindra': ['Thar', 'Scorpio', 'XUV500', 'Bolero', 'TUV300'],
    'BMW': ['3 Series', 'X1', 'X5', '5 Series', '7 Series'],
    'Audi': ['A4', 'A6', 'Q7', 'A8'],
    'Mercedes-Benz': ['C-Class', 'E-Class', 'S-Class', 'GLA', 'GLC'],
    'Skoda': ['Octavia', 'Superb', 'Kodiaq', 'Rapid', 'Karoq'],
    'Renault': ['Kwid', 'Duster', 'Captur', 'Triber', 'Kiger'],
    'Kia': ['Seltos', 'Sonet', 'Carnival', 'Sportage', 'EV6'],
    'Nissan': ['Magnite', 'Kicks', 'Sunny', 'Terrano', 'GT-R'],
    'Volvo': ['XC40', 'XC60', 'XC90', 'S60', 'S90'],
    'Bentley': ['Continental', 'Bentayga', 'Flying Spur'],
    'Datsun': ['Go', 'Go+', 'Redi-Go'],
    'Ferrari': ['488', 'Portofino', 'F8 Tributo'],
    'Force': ['Gurkha', 'Trax'],
    'Isuzu': ['D-Max', 'MU-X'],
    'Jaguar': ['XE', 'XF', 'F-Pace'],
    'Jeep': ['Compass', 'Wrangler', 'Grand Cherokee'],
    'Land Rover': ['Discovery', 'Range Rover', 'Defender'],
    'Lexus': ['ES', 'NX', 'RX'],
    'Maserati': ['Ghibli', 'Levante', 'Quattroporte'],
    'Mercedes-AMG': ['AMG C63', 'AMG GLE', 'AMG GT'],
    'MG': ['Hector', 'ZS EV', 'Gloster'],
    'Mini': ['Cooper', 'Countryman', 'Clubman'],
    'Porsche': ['911', 'Cayenne', 'Macan'],
    'Rolls-Royce': ['Phantom', 'Ghost', 'Cullinan']
}

# Normalize brand names (handle whitespace and case)
def normalize_brand(brand):
    return brand.strip().lower()  # Remove spaces and normalize case

# Precompute valid brands and models
valid_models_dict = {}
valid_brands = []
for brand in brand_model_mapping.keys():
    normalized_brand = normalize_brand(brand)
    # Check if brand exists in encoder (case-insensitive)
    if normalized_brand in [normalize_brand(c) for c in le_brand.classes_]:
        # Get models that exist in le_model.classes_
        valid_models = sorted([m for m in brand_model_mapping[brand] if m in le_model.classes_])
        if valid_models:  # Only include brands with at least one valid model
            valid_brands.append(brand)
            valid_models_dict[brand] = valid_models

if not valid_brands:
    st.error("No valid brands with compatible models found in the label encoder. Please retrain the model.")
    st.stop()

# Extract valid categories from preprocessor
def get_preprocessor_categories(preprocessor):
    categories = {}
    for name, transformer, columns in preprocessor.transformers_:
        if hasattr(transformer, 'categories_'):
            for col, cats in zip(columns, transformer.categories_):
                categories[col] = list(cats)
    return categories

# Assume preprocessor is a ColumnTransformer with OneHotEncoder for categorical features
try:
    preprocessor_categories = get_preprocessor_categories(preprocessor)
    valid_seller_types = preprocessor_categories.get('seller_type', ["Individual", "Dealer", "Trustmark Dealer"])
    valid_fuel_types = preprocessor_categories.get('fuel_type', ["Petrol", "Diesel", "CNG", "LPG", "Electric"])
    valid_transmission_types = preprocessor_categories.get('transmission_type', ["Manual", "Automatic"])
except:
    # Fallback defaults if preprocessor categories can't be extracted
    valid_seller_types = ["Individual", "Dealer", "Trustmark Dealer"]
    valid_fuel_types = ["Petrol", "Diesel", "CNG", "LPG", "Electric"]
    valid_transmission_types = ["Manual", "Automatic"]

# Initialize session state
if 'selected_brand' not in st.session_state:
    st.session_state.selected_brand = valid_brands[0]
if 'selected_model' not in st.session_state:
    st.session_state.selected_model = valid_models_dict[valid_brands[0]][0]

# Streamlit app
st.title('🚗 CarWorth: Used Car Price Prediction')

# Add the About section to the sidebar
st.sidebar.markdown("""
### ℹ️ About This Prediction Model

**Model Purpose:**  
Predicts the fair market value of used cars in India based on key specifications.


**Key Features Considered:**
- Brand & Model
- Vehicle Age
- Kilometers Driven
- Fuel & Transmission Type
- Engine Specifications
- Mileage
- Seller Type

**Limitations:**
1. Works best for cars 1-15 years old
2. Predictions are estimates only
3. Regional price variations may apply

**Note:**  
Actual market prices may vary based on condition, location, and market demand.
""")

st.markdown("""
Predict the market price of your used car based on its specifications.
This model was trained on data from cardekho.com in India.
""")

# Brand and model selection
brand = st.selectbox(
    "Brand",
    valid_brands,
    key="brand_select",
    help="Select the car brand."
)

# Update session state when brand changes
if normalize_brand(brand) != normalize_brand(st.session_state.selected_brand):
    st.session_state.selected_brand = brand
    st.session_state.selected_model = valid_models_dict[brand][0]  # Default to first valid model

# Model selection
model_name = st.selectbox(
    "Model",
    valid_models_dict[st.session_state.selected_brand],
    index=valid_models_dict[st.session_state.selected_brand].index(st.session_state.selected_model),
    key="model_select",
    help="Select the car model."
)

# Update session state for model
st.session_state.selected_model = model_name

# Create form for other inputs
with st.form("car_details"):
    st.header("Car Specifications")
    
    # Vehicle specifications
    col1, col2 = st.columns(2)
    with col1:
        vehicle_age = st.number_input("Vehicle Age (years)", min_value=1, max_value=30, value=5, help="Enter the age of the vehicle in years.")
    with col2:
        km_driven = st.number_input("Kilometers Driven", min_value=0, max_value=500000, value=50000, help="Enter the total kilometers driven.")
    
    col3, col4 = st.columns(2)
    with col3:
        seller_type = st.selectbox("Seller Type", valid_seller_types, help="Select the type of seller.")
    with col4:
        fuel_type = st.selectbox("Fuel Type", valid_fuel_types, help="Select the fuel type.")
    
    col5, col6 = st.columns(2)
    with col5:
        transmission_type = st.selectbox("Transmission Type", valid_transmission_types, help="Select the transmission type.")
    with col6:
        seats = st.selectbox("Number of Seats", [2, 4, 5, 6, 7, 8, 9], help="Select the number of seats.")
    
    col7, col8, col9 = st.columns(3)
    with col7:
        mileage = st.number_input("Mileage (kmpl or km/kg)", min_value=5.0, max_value=50.0, value=15.0, step=0.1, help="Enter the mileage in kmpl or km/kg.")
    with col8:
        engine = st.number_input("Engine Capacity (cc)", min_value=500, max_value=5000, value=1200, help="Enter the engine capacity in cubic centimeters (cc).")
    with col9:
        max_power = st.number_input("Max Power (bhp)", min_value=10.0, max_value=500.0, value=80.0, step=0.1, help="Enter the maximum power in brake horsepower (bhp).")
    
    # Submit button
    submitted = st.form_submit_button("Estimate Price", help="Click to predict the car price.")

# Handle form submission
if submitted:
    # Input validation
    if mileage <= 0:
        st.error("Mileage must be greater than 0.")
    elif engine <= 0:
        st.error("Engine capacity must be greater than 0.")
    elif max_power <= 0:
        st.error("Max power must be greater than 0.")
    else:
        try:
            # Find the original brand name in le_brand.classes_ that matches normalized_brand
            normalized_brand = normalize_brand(st.session_state.selected_brand)
            original_brand = next(c for c in le_brand.classes_ if normalize_brand(c) == normalized_brand)
            
            # Create input DataFrame
            input_data = pd.DataFrame({
                'brand': [original_brand],
                'model': [model_name],
                'vehicle_age': [vehicle_age],
                'km_driven': [km_driven],
                'seller_type': [seller_type],
                'fuel_type': [fuel_type],
                'transmission_type': [transmission_type],
                'mileage': [mileage],
                'engine': [engine],
                'max_power': [max_power],
                'seats': [seats]
            })
            
            # Transform categorical features
            input_data['brand'] = le_brand.transform(input_data['brand'])
            input_data['model'] = le_model.transform(input_data['model'])
            
            # Preprocess and predict
            with st.spinner("Predicting..."):
                input_transformed = preprocessor.transform(input_data)
                predicted_price = model.predict(input_transformed)[0]
            
            # Display result
            st.success(f"### Estimated Price: ₹{predicted_price:,.2f}")
            
        except ValueError as ve:
            st.error(f"Input error: {str(ve)}. Please ensure all inputs are valid.")
        except Exception as e:
            st.error(f"An unexpected error occurred: {str(e)}. Please try again or contact support.")
