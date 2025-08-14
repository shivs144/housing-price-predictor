
from datetime import timedelta
import pandas as pd
import numpy as np
from catboost import CatBoostRegressor
from sklearn.cluster import KMeans
from sklearn.preprocessing import LabelEncoder, StandardScaler, OneHotEncoder
import matplotlib.pyplot as plt
import seaborn as sns


df = pd.read_csv('Makaan_Properties_Buy.csv', encoding='cp1252')
# print(df.columns)

# Size Transformation
df['Size_Clean'] = (
    df['Size']
    .str.lower()
    .str.replace('sq ft', '', regex=False)
    .str.replace(',', '', regex=False)
    .str.strip()
)

# Delete irrelevant fields
df = df.drop(['Size','Property_id' , 'Property_Name', 'Builder_name','is_commercial_Listing' ,'builder_id', 'City_id', 'Locality_ID', 'Sub_urban_ID', 'listing_domain_score', 'Listing_Category',  'Project_URL'], axis=1)

df['Size_Clean'] = pd.to_numeric(df['Size_Clean'], errors='coerce')
df.rename(columns={'Size_Clean': 'Area'}, inplace=True)

# Price Transformation
df['Price'] = df['Price'].str.replace(',','', regex=False).str.strip()
df['Price'] = pd.to_numeric(df['Price'], errors='coerce')

# Price Per Unit Area Transformation
df['Price_per_unit_area'] = df['Price_per_unit_area'].str.replace(',', '').str.strip()
df['Price_per_unit_area'] = pd.to_numeric(df['Price_per_unit_area'], errors='coerce')
# print(df['Price_per_unit_area'].sum())

# # Posted On Transformation
Posted_Reference_Date = pd.to_datetime("2022-07-25")

def PostedTransformation(posted):
    posted = str(posted).lower()
    if 'days' or 'day' in posted:
        days = int(posted.split()[0]) if posted[0].isdigit() else 1
        return Posted_Reference_Date - timedelta(days=days)
    elif 'months' or 'month' in posted:
        months = int(posted.split()[0]) if posted[0].isdigit() else 1
        return Posted_Reference_Date - pd.DateOffset(months=months)
    elif "minute" or 'minutes' in posted:
        minutes = int(posted.split()[0]) if posted[0].isdigit() else 1
        return Posted_Reference_Date - timedelta(minutes=minutes)
    elif "hour" or 'hours' in posted:
        hours = int(posted.split()[0]) if posted[0].isdigit() else 1
        return Posted_Reference_Date - timedelta(hours=hours)
    else:
        return 0

df['Posted_On'] = df['Posted_On'].apply(PostedTransformation)
# print(df['Posted_On'])

# No of BHK Transformation
df['No_of_BHK'] = df['No_of_BHK'].str.split(' ').str[0]
df['No_of_BHK'] = pd.to_numeric(df['No_of_BHK'], errors='coerce')
# print(df['No_of_BHK'].max())

# print(df['Builder_name'].isnull().sum())

# Handle NA in Description Field
df['description'] = df['description'].fillna('NA')
df['description'] = df['description'].replace('NA', 'Gated community with 24/7 security, Landscaped garden & terrace area, Spacious bedrooms with balconies, Premium modular kitchen fittings, High-quality construction and vastu-compliant layout,Proximity to SG Highway, ISKCON, and major business hubs')
# print(df['description'].isnull().sum())

# Handle NA in Locality Name
df['Locality_Name'] = df['Locality_Name'].fillna('NA')
df['Locality_Name'] = df['Locality_Name'].replace('NA', 'Godavari')
# print(df['Locality_Name'].isnull().sum())

# NA Handling in Property Status
# print(df['Property_status'].value_counts(normalize=True))
property_values = ['Ready to move', 'Under Construction']
weights = [ 0.661871, 0.338129]

blank = df['Property_status'].isnull() | (df['Property_status'] == '')
df.loc[blank, 'Property_status'] = np.random.choice(property_values, size = blank.sum(),p = weights)

# Label Encoding
encoder = LabelEncoder()
encode_columns = ['City_name', 'Property_type', 'No_of_BHK','Property_building_status', 'Locality_Name', 'Sub_urban_name', 'is_furnished', 'Property_status']

for col in encode_columns:
    df[col] = encoder.fit_transform(df[col])

# Standard Scaling
scaling = StandardScaler()
scaling_columns = ['Price_per_unit_area', 'Area']

for column in scaling_columns:
    df[column] = scaling.fit_transform(df[[column]])

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
# print(df.columns)
