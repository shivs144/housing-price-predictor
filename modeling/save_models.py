import pandas as pd
import numpy as np
from catboost import CatBoostRegressor
from sklearn.cluster import KMeans
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
import pickle
import os

# --- ETL (adapted from etl/ETL.py) ---
df = pd.read_csv('../data/Makaan_Properties_Buy.csv', encoding='cp1252')

# Size Transformation
df['Size_Clean'] = (
    df['Size']
    .str.lower()
    .str.replace('sq ft', '', regex=False)
    .str.replace(',', '', regex=False)
    .str.strip()
)
df = df.drop(['Size','Property_id' , 'Property_Name', 'Builder_name','is_commercial_Listing' ,'builder_id', 'City_id', 'Locality_ID', 'Sub_urban_ID', 'listing_domain_score', 'Listing_Category',  'Project_URL'], axis=1)
df['Size_Clean'] = pd.to_numeric(df['Size_Clean'], errors='coerce')
df.rename(columns={'Size_Clean': 'Area'}, inplace=True)

# Price Transformation
df['Price'] = df['Price'].str.replace(',','', regex=False).str.strip()
df['Price'] = pd.to_numeric(df['Price'], errors='coerce')

# Price Per Unit Area Transformation
df['Price_per_unit_area'] = df['Price_per_unit_area'].str.replace(',', '').str.strip()
df['Price_per_unit_area'] = pd.to_numeric(df['Price_per_unit_area'], errors='coerce')

# No of BHK Transformation
df['No_of_BHK'] = df['No_of_BHK'].str.split(' ').str[0]
df['No_of_BHK'] = pd.to_numeric(df['No_of_BHK'], errors='coerce')

# Handle NA in Description Field
df['description'] = df['description'].fillna('NA')
df['description'] = df['description'].replace('NA', 'Gated community with 24/7 security, Landscaped garden & terrace area, Spacious bedrooms with balconies, Premium modular kitchen fittings, High-quality construction and vastu-compliant layout,Proximity to SG Highway, ISKCON, and major business hubs')

# Handle NA in Locality Name
df['Locality_Name'] = df['Locality_Name'].fillna('NA')
df['Locality_Name'] = df['Locality_Name'].replace('NA', 'Godavari')

# NA Handling in Property Status
property_values = ['Ready to move', 'Under Construction']
weights = [ 0.661871, 0.338129]
blank = df['Property_status'].isnull() | (df['Property_status'] == '')
df.loc[blank, 'Property_status'] = np.random.choice(property_values, size = blank.sum(),p = weights)

# Store original data for encoding mapping
original_data = df.copy()

# Label Encoding
encoders = {}
encode_columns = ['City_name', 'Property_type', 'No_of_BHK','Property_building_status', 'Locality_Name', 'Sub_urban_name', 'is_furnished', 'Property_status']
for col in encode_columns:
    encoder = LabelEncoder()
    df[col] = encoder.fit_transform(df[col])
    encoders[col] = encoder

# Standard Scaling
scalers = {}
scaling_columns = ['Price_per_unit_area', 'Area']
for column in scaling_columns:
    scaler = StandardScaler()
    df[column] = scaler.fit_transform(df[[column]])
    scalers[column] = scaler

# Handle Amenities and encoding
amenities = [
    "car parking", "full power backup", "banquet hall", "gymnasium", "landscape garden",
    "tree planting", "children's play area", "24x7 water supply", "indoor games", "fire fighting system",
    "cctv", "club house", "terrace garden", "jogging track", "swimming pool", "library",
    "vehicle free zone", "waiting lounge", "senior citizen area", "cricket pitch", "society office",
    "solar lighting", "internal roads", "yoga/meditation area", "tennis court"
]
df['description'] = df['description'].str.lower()
df['description'] = df['description'].fillna('')
for amenity in amenities:
    col_name = amenity.replace(" ", "").replace("/", "").replace("'", "").replace("-", "_")
    df[col_name] = df['description'].apply(lambda x: 1 if amenity in x else 0)
df = df.drop(['description'],axis= 1)
cols_to_encode = ['is_plot', 'is_RERA_registered', 'is_Apartment',
                  'is_ready_to_move', 'is_PentaHouse', 'is_studio']
df[cols_to_encode] = df[cols_to_encode].astype(int)

# --- Modeling (adapted from modeling/Modeling.py) ---
x = df[['Area','City_name', 'Locality_Name', 'Sub_urban_name', 'Property_type', 'No_of_BHK', 'is_furnished','Property_building_status', 'is_ready_to_move', 'is_PentaHouse' ,'is_Apartment','Property_status']]
y = df['Price']

x_train, x_test, y_train, y_test = train_test_split(x, y, random_state=42, test_size=0.2)

catboost_model = CatBoostRegressor(
    iterations=6000,
    learning_rate=0.1,
    depth=6,
    eval_metric='R2',
    verbose=100,
    random_state=42
)
catboost_model.fit(x_train, y_train, eval_set=(x_test, y_test), early_stopping_rounds=200)

# Save CatBoost model
os.makedirs('../models', exist_ok=True)
pickle.dump(catboost_model, open('../models/cat_boost.pkl', 'wb'))

# Save encoders and scalers
pickle.dump(encoders, open('../models/encoders.pkl', 'wb'))
pickle.dump(scalers, open('../models/scalers.pkl', 'wb'))

# Predict and cluster
y_pred = catboost_model.predict(x_test)
residual = y_test - y_pred
cluster = pd.DataFrame(x_test, columns=x.columns)
cluster['Residual'] = residual.values
cluster['Predicted_Price'] = y_pred
cluster['Actual_Price'] = y_test.values

kmeans = KMeans(n_clusters=3, random_state=42)
cluster['cluster'] = kmeans.fit_predict(cluster[['Residual', 'Predicted_Price']])
pickle.dump(kmeans, open('../models/Clustering.pkl', 'wb'))

print('Models saved to ../models/cat_boost.pkl and ../models/Clustering.pkl')
print('Encoders and scalers saved to ../models/encoders.pkl and ../models/scalers.pkl')