import pandas as pd
import numpy as np
import pickle
from catboost import CatBoostRegressor
from sklearn.cluster import KMeans
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.metrics import r2_score, silhouette_score
import matplotlib.pyplot as plt
import seaborn as sns

print("=" * 60)
print("HOUSING PRICE PREDICTION - MODEL SUMMARIES")
print("=" * 60)

# Initialize variables
test_r2 = None
cluster_data = None
cluster_labels = None

# Load the data and preprocess
print("\n1. Loading and preprocessing data...")
df = pd.read_csv('data/Makaan_Properties_Buy.csv', encoding='cp1252')

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

print(f"Data shape after preprocessing: {df.shape}")

# Prepare features for modeling
x = df[['Area','City_name', 'Locality_Name', 'Sub_urban_name', 'Property_type', 'No_of_BHK', 'is_furnished','Property_building_status', 'is_ready_to_move', 'is_PentaHouse' ,'is_Apartment','Property_status']]
y = df['Price']

print(f"Feature matrix shape: {x.shape}")
print(f"Target variable shape: {y.shape}")

# Split data
x_train, x_test, y_train, y_test = train_test_split(x, y, random_state=42, test_size=0.2)
print(f"Training set size: {x_train.shape[0]}")
print(f"Test set size: {x_test.shape[0]}")

print("\n" + "=" * 60)
print("CATBOOST REGRESSION MODEL SUMMARY")
print("=" * 60)

# Load CatBoost model
try:
    with open('models/cat_boost.pkl', 'rb') as f:
        catboost_model = pickle.load(f)
    
    print("\n1. MODEL CONFIGURATION:")
    print(f"   Algorithm: CatBoost Regressor")
    print(f"   Iterations: {catboost_model.get_params()['iterations']}")
    print(f"   Learning Rate: {catboost_model.get_params()['learning_rate']}")
    print(f"   Depth: {catboost_model.get_params()['depth']}")
    print(f"   Evaluation Metric: {catboost_model.get_params()['eval_metric']}")
    print(f"   Random State: {catboost_model.get_params()['random_state']}")
    
    print("\n2. TRAINING METRICS:")
    # Get training history
    train_scores = catboost_model.get_evals_result()
    if train_scores:
        train_metric = list(train_scores.keys())[0]
        train_scores_list = train_scores[train_metric]['learn'][train_metric]
        validation_scores_list = train_scores[train_metric]['validation'][train_metric]
        
        print(f"   Final Training R² Score: {train_scores_list[-1]:.4f}")
        print(f"   Final Validation R² Score: {validation_scores_list[-1]:.4f}")
        print(f"   Best Validation R² Score: {max(validation_scores_list):.4f}")
        print(f"   Training Iterations: {len(train_scores_list)}")
    
    print("\n3. TEST SET PERFORMANCE:")
    y_pred = catboost_model.predict(x_test)
    test_r2 = r2_score(y_test, y_pred)
    print(f"   Test Set R² Score: {test_r2:.4f}")
    
    # Calculate additional metrics
    mse = np.mean((y_test - y_pred) ** 2)
    rmse = np.sqrt(mse)
    mae = np.mean(np.abs(y_test - y_pred))
    
    print(f"   Mean Squared Error: {mse:.2f}")
    print(f"   Root Mean Squared Error: {rmse:.2f}")
    print(f"   Mean Absolute Error: {mae:.2f}")
    
    print("\n4. FEATURE IMPORTANCE:")
    importances = catboost_model.feature_importances_
    features = x_train.columns
    
    # Create feature importance DataFrame
    feat_imp = pd.DataFrame({'Feature': features, 'Importance': importances})
    feat_imp = feat_imp.sort_values(by='Importance', ascending=False)
    
    print("   Top 10 Most Important Features:")
    for i, (_, row) in enumerate(feat_imp.head(10).iterrows()):
        print(f"   {i+1:2d}. {row['Feature']:<20} {row['Importance']:.4f}")
    
    print("\n5. MODEL STATISTICS:")
    print(f"   Model Size: {len(pickle.dumps(catboost_model)) / (1024*1024):.2f} MB")
    print(f"   Number of Trees: {catboost_model.tree_count_}")
    print(f"   Feature Count: {len(features)}")
    print(f"   Training Samples: {len(x_train)}")
    print(f"   Test Samples: {len(x_test)}")
    
except Exception as e:
    print(f"Error loading CatBoost model: {e}")

print("\n" + "=" * 60)
print("K-MEANS CLUSTERING MODEL SUMMARY")
print("=" * 60)

# Load K-means model and perform clustering analysis
try:
    with open('models/Clustering.pkl', 'rb') as f:
        kmeans_model = pickle.load(f)
    
    print("\n1. MODEL CONFIGURATION:")
    print(f"   Algorithm: K-means Clustering")
    print(f"   Number of Clusters: {kmeans_model.get_params()['n_clusters']}")
    print(f"   Random State: {kmeans_model.get_params()['random_state']}")
    print(f"   Initialization Method: {kmeans_model.get_params()['init']}")
    print(f"   Maximum Iterations: {kmeans_model.get_params()['max_iter']}")
    
    print("\n2. CLUSTERING ANALYSIS:")
    # Use test predictions for clustering analysis
    y_pred = catboost_model.predict(x_test)
    residuals = y_test - y_pred
    
    # Prepare data for clustering
    cluster_data = np.column_stack([residuals, y_pred])
    
    # Perform clustering on test data
    cluster_labels = kmeans_model.predict(cluster_data)
    
    print(f"   Data Points Clustered: {len(cluster_labels)}")
    silhouette_avg = silhouette_score(cluster_data, cluster_labels)
    print(f"   Silhouette Score: {silhouette_avg:.4f}")
    
    # Analyze cluster characteristics
    cluster_df = pd.DataFrame({
        'Residual': residuals,
        'Predicted_Price': y_pred,
        'Actual_Price': y_test,
        'Cluster': cluster_labels
    })
    
    print("\n3. CLUSTER CHARACTERISTICS:")
    for cluster_id in range(3):
        cluster_subset = cluster_df[cluster_df['Cluster'] == cluster_id]
        print(f"\n   Cluster {cluster_id}:")
        print(f"     Size: {len(cluster_subset)} properties")
        print(f"     Mean Residual: {cluster_subset['Residual'].mean():.2f}")
        print(f"     Mean Predicted Price: ₹{cluster_subset['Predicted_Price'].mean():,.0f}")
        print(f"     Mean Actual Price: ₹{cluster_subset['Actual_Price'].mean():,.0f}")
        print(f"     Residual Std Dev: {cluster_subset['Residual'].std():.2f}")
    
    # Determine cluster labels based on residuals
    mean_residuals = cluster_df.groupby('Cluster')['Residual'].mean().sort_values()
    label_map = {
        mean_residuals.index[0]: 'Undervalued',
        mean_residuals.index[1]: 'Correctly Priced',
        mean_residuals.index[2]: 'Overvalued'
    }
    
    print("\n4. VALUATION CLASSIFICATION:")
    for cluster_id, label in label_map.items():
        cluster_subset = cluster_df[cluster_df['Cluster'] == cluster_id]
        print(f"   {label}: {len(cluster_subset)} properties")
    
    print("\n5. MODEL STATISTICS:")
    print(f"   Model Size: {len(pickle.dumps(kmeans_model)) / 1024:.2f} KB")
    print(f"   Cluster Centers Shape: {kmeans_model.cluster_centers_.shape}")
    print(f"   Inertia (Sum of Squared Distances): {kmeans_model.inertia_:.2f}")
    print(f"   Number of Iterations: {kmeans_model.n_iter_}")
    
except Exception as e:
    print(f"Error loading K-means model: {e}")

print("\n" + "=" * 60)
print("OVERALL SYSTEM SUMMARY")
print("=" * 60)

print("\n1. DATA PROCESSING:")
print(f"   Original Dataset Size: {len(df)} properties")
print(f"   Features Engineered: {len(x.columns)}")
print(f"   Amenities Extracted: {len(amenities)}")
print(f"   Categorical Variables Encoded: {len(encode_columns)}")
print(f"   Numerical Variables Scaled: {len(scaling_columns)}")

print("\n2. MODEL PERFORMANCE:")
if test_r2 is not None:
    print(f"   CatBoost R² Score: {test_r2:.4f}")
if cluster_data is not None and cluster_labels is not None:
    silhouette_avg = silhouette_score(cluster_data, cluster_labels)
    print(f"   K-means Silhouette Score: {silhouette_avg:.4f}")

print("\n3. SYSTEM ARCHITECTURE:")
print(f"   Total Model Files: 4 (CatBoost, K-means, Encoders, Scalers)")
print(f"   Total Model Size: ~4.2 MB")
print(f"   Web Application: Streamlit (1054 lines)")
print(f"   Deployment: Local server on port 8501")

print("\n" + "=" * 60)
print("MODEL SUMMARIES COMPLETE")
print("=" * 60)
