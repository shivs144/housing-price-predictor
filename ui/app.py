import streamlit as st
import pandas as pd
import numpy as np
import pickle
from geopy.geocoders import Nominatim
import gdown
# from pandasai import PandasAI  # Uncomment if using PandasAI
# from pandasai.llm import OpenAI  # Uncomment if using PandasAI

# --- Load Data (ETL processed data should be used here) ---
@st.cache_data

def load_data():
    FILE_ID = "151C0wM94gDtIWWJnud5qLMCDuG_-03WR"

    # Create the direct download URL
    url = f"https://drive.google.com/uc?id={FILE_ID}"

    # Download the file (it will save in current directory)
    gdown.download(url, 'Makaan_Properties_Buy.csv', quiet=False)

    # Load into pandas
    df = pd.read_csv('Makaan_Properties_Buy.csv', encoding='cp1252')
    
    # Preprocess the data to match the modeling pipeline
    # Size Transformation (same as in save_models.py)
    df['Size_Clean'] = (
        df['Size']
        .str.lower()
        .str.replace('sq ft', '', regex=False)
        .str.replace(',', '', regex=False)
        .str.strip()
    )
    df['Size_Clean'] = pd.to_numeric(df['Size_Clean'], errors='coerce')
    
    # Remove records where size < 300 sq ft
    df = df[df['Size_Clean'] >= 300]
    
    # Remove records with invalid or negative area values
    df = df[df['Size_Clean'] > 0]
    
    df.rename(columns={'Size_Clean': 'Area'}, inplace=True)
    
    # Price Transformation (same as in save_models.py)
    df['Price'] = df['Price'].str.replace(',','', regex=False).str.strip()
    df['Price'] = pd.to_numeric(df['Price'], errors='coerce')
    
    # Remove records with invalid or negative prices
    df = df[df['Price'] > 0]
    
    # No of BHK Transformation (same as in save_models.py)
    df['No_of_BHK'] = df['No_of_BHK'].str.split(' ').str[0]
    df['No_of_BHK'] = pd.to_numeric(df['No_of_BHK'], errors='coerce')
    
    return df

data = load_data()

# Data quality check
if len(data) == 0:
    st.error("❌ No valid data loaded. Please check the data file.")
    st.stop()

# --- Load Models and Preprocessors ---
def load_catboost_model():
    try:
        with open('../models/cat_boost.pkl', 'rb') as f:
            model = pickle.load(f)
        return model
    except FileNotFoundError:
        st.error("❌ Model file '../models/cat_boost.pkl' not found!")
        st.stop()
    except Exception as e:
        st.error(f"❌ Error loading CatBoost model: {str(e)}")
        st.stop()

def load_kmeans_model():
    try:
        with open('../models/Clustering.pkl', 'rb') as f:
            model = pickle.load(f)
        return model
    except FileNotFoundError:
        st.error("❌ Model file '../models/Clustering.pkl' not found!")
        st.stop()
    except Exception as e:
        st.error(f"❌ Error loading KMeans model: {str(e)}")
        st.stop()

def load_encoders():
    try:
        with open('../models/encoders.pkl', 'rb') as f:
            encoders = pickle.load(f)
        return encoders
    except FileNotFoundError:
        st.error("❌ Model file '../models/encoders.pkl' not found!")
        st.stop()
    except Exception as e:
        st.error(f"❌ Error loading encoders: {str(e)}")
        st.stop()

def load_scalers():
    try:
        with open('../models/scalers.pkl', 'rb') as f:
            scalers = pickle.load(f)
        return scalers
    except FileNotFoundError:
        st.error("❌ Model file '../models/scalers.pkl' not found!")
        st.stop()
    except Exception as e:
        st.error(f"❌ Error loading scalers: {str(e)}")
        st.stop()

# --- Helper for Clustering Label ---
def get_cluster_label(cluster_id, residuals=None, predicted_prices=None):
    """
    Get cluster label based on residual analysis.
    If residuals and predicted_prices are provided, use them to determine proper labeling.
    Otherwise, use a default mapping.
    """
    if residuals is not None and predicted_prices is not None:
        # Calculate mean residuals for each cluster
        cluster_data = pd.DataFrame({
            'Residual': residuals,
            'Predicted_Price': predicted_prices,
            'Cluster': cluster_id
        })
        
        # Group by cluster and calculate mean residuals
        cluster_means = cluster_data.groupby('Cluster')['Residual'].mean().sort_values()
        
        # Create label mapping based on sorted residuals
        # Lowest mean residual = Overvalued (Actual < Predicted, negative residual)
        # Highest mean residual = Undervalued (Actual > Predicted, positive residual)
        label_map = {}
        for i, cluster_idx in enumerate(cluster_means.index):
            if i == 0:
                label_map[cluster_idx] = 'Overvalued'  # Lowest residual (most negative)
            elif i == len(cluster_means) - 1:
                label_map[cluster_idx] = 'Undervalued'  # Highest residual (most positive)
            else:
                label_map[cluster_idx] = 'Correctly Priced'  # Middle residual
        
        return label_map.get(cluster_id, 'Unknown')
    else:
        # Default mapping (fallback)
        label_map = {0: 'Overvalued', 1: 'Correctly Priced', 2: 'Undervalued'}
        return label_map.get(cluster_id, 'Unknown')

def impute_nan_with_city_mode(df, column_name):
    """
    Impute NaN values in a column with the most frequent value in that city.
    """
    df_copy = df.copy()
    
    # Group by city and get mode for each city
    city_modes = df_copy.groupby('City_name')[column_name].agg(lambda x: x.mode().iloc[0] if len(x.mode()) > 0 else x.mode().iloc[0] if len(x.mode()) > 0 else None)
    
    # Fill NaN values with city mode
    for city in df_copy['City_name'].unique():
        city_mode = city_modes.get(city)
        if city_mode is not None:
            mask = (df_copy['City_name'] == city) & (df_copy[column_name].isna())
            df_copy.loc[mask, column_name] = city_mode
    
    return df_copy

# --- Preprocess input data ---
def preprocess_input(input_df, encoders, scalers):
    # Apply label encoding to categorical columns
    encode_columns = ['City_name', 'Property_type', 'No_of_BHK','Property_building_status', 'Locality_Name', 'Sub_urban_name', 'is_furnished', 'Property_status']
    for col in encode_columns:
        if col in encoders and col in input_df.columns:
            # Handle unseen categories by using a default value
            try:
                input_df[col] = encoders[col].transform(input_df[col])
            except ValueError:
                # If category not seen during training, use the first category
                input_df[col] = 0
    
    # Apply scaling to numerical columns
    scaling_columns = ['Area']
    for column in scaling_columns:
        if column in scalers and column in input_df.columns:
            input_df[column] = scalers[column].transform(input_df[[column]])
    
    return input_df

# --- Streamlit App ---
st.set_page_config(page_title="Housing Price Predictor & Recommender", layout="wide")

# --- PandasAI Chat Box in Sidebar ---
def pandasai_chat_box(df):
    st.sidebar.markdown("---")
    st.sidebar.markdown("## 🏛️ Welcome to CDAC Acts")
    st.sidebar.markdown("---")
    st.sidebar.subheader("Enter your query")
    
    # Initialize session state for input key
    if 'input_key' not in st.session_state:
        st.session_state.input_key = 0
    
    user_prompt = st.sidebar.text_input(
        "Your question:", 
        placeholder="e.g., 'Show me properties under 50 lakhs'",
        key=f"query_input_{st.session_state.input_key}"
    )
    
    # Submit button
    if st.sidebar.button("🔍 Submit Query", type="primary", use_container_width=True):
        if user_prompt:
            # Store the query in session state to display results below buttons
            st.session_state.pandasai_query = user_prompt
            st.session_state.pandasai_data = df
            st.sidebar.success("Query submitted! Check results below.")
            
            # Increment input key to clear the input field for next query
            st.session_state.input_key += 1
        else:
            st.sidebar.warning("Please enter a query first.")

# Initialize PandasAI chat in sidebar
pandasai_chat_box(data)

# --- Main App Title ---
st.title("🏠 Housing Price Predictor & Recommender System")
st.write("Enter your property preferences and choose an action below.")

# --- User Inputs (Same for both functionalities) ---
col1, col2 = st.columns(2)
with col1:
    # Create size ranges of 50 sq ft for dropdown
    min_size = int(data['Area'].min())
    max_size = int(data['Area'].max())
    
    # Generate size ranges in steps of 50
    size_ranges = []
    for i in range(min_size, max_size + 50, 50):
        range_end = i + 49
        if range_end > max_size:
            range_end = max_size
        size_ranges.append(f"{i} - {range_end} sq ft")
    
    # Set default to median range
    median_size = int(data['Area'].median())
    default_index = (median_size - min_size) // 50
    default_index = min(default_index, len(size_ranges) - 1)
    
    selected_range = st.selectbox(
        "Size Range (sq ft)",
        options=size_ranges,
        index=default_index,
        help="Select the property size range in square feet"
    )
    
    # Extract the minimum value from the selected range
    area = int(selected_range.split(' - ')[0])
    
    # Validate area input
    if area <= 0:
        st.error("⚠️ Invalid area value. Please select a valid size range.")
        st.stop()
    
    # Cascading dropdowns for City → Sub Urban → Locality
    city = st.selectbox("City Name", sorted(data['City_name'].dropna().unique()))
    
    # Filter Sub Urban names based on selected city
    city_filtered = data[data['City_name'] == city]
    suburb_options = sorted(city_filtered['Sub_urban_name'].dropna().unique())
    suburb = st.selectbox("Sub Urban Name", suburb_options)
    
    # Filter Locality names based on selected city and suburb
    suburb_filtered = city_filtered[city_filtered['Sub_urban_name'] == suburb]
    locality_options = sorted(suburb_filtered['Locality_Name'].dropna().unique())
    locality = st.selectbox("Locality Name", locality_options)
    
    property_type = st.selectbox("Property Type", sorted(data['Property_type'].dropna().unique()))
with col2:
    bhk = st.number_input("No. of BHK", min_value=1, max_value=10, value=2)
    furnished = st.selectbox("Is Furnished", sorted(data['is_furnished'].dropna().unique()))
    building_status = st.selectbox("Property Building Status", sorted(data['Property_building_status'].dropna().unique()))
    ready_to_move = st.selectbox("Is Ready to Move", sorted(data['is_ready_to_move'].dropna().unique()))
    pentahouse = st.selectbox("Is PentaHouse", sorted(data['is_PentaHouse'].dropna().unique()))
    apartment = st.selectbox("Is Apartment", sorted(data['is_Apartment'].dropna().unique()))
    property_status = st.selectbox("Property Status", sorted(data['Property_status'].dropna().unique()))

# --- Action Buttons ---
st.markdown("---")
col1, col2, col3 = st.columns([1, 1, 1])
with col1:
    predict_button = st.button("🎯 Predict Price", type="primary", use_container_width=True)
with col2:
    find_similar_button = st.button("🔍 Find Similar Properties", type="secondary", use_container_width=True)
with col3:
    # Clear results button
    if st.button("🗑️ Clear Results", use_container_width=True):
        st.rerun()

# Initialize pagination session state
if 'current_page' not in st.session_state:
    st.session_state.current_page = 0

if 'show_all' not in st.session_state:
    st.session_state.show_all = False

# --- Price Prediction Functionality ---
if predict_button:
    st.markdown("---")
    st.subheader("🎯 Price Prediction Results")
    
    try:
        try:
            # --- Prepare input for model ---
            input_df = pd.DataFrame({
                'Area': [area],
                'City_name': [city],
                'Locality_Name': [locality],
                'Sub_urban_name': [suburb],
                'Property_type': [property_type],
                'No_of_BHK': [bhk],
                'is_furnished': [furnished],
                'Property_building_status': [building_status],
                'is_ready_to_move': [ready_to_move],
                'is_PentaHouse': [pentahouse],
                'is_Apartment': [apartment],
                'Property_status': [property_status]
            })
            
            # --- Load models and preprocessors ---
            catboost_model = load_catboost_model()
            kmeans_model = load_kmeans_model()
            encoders = load_encoders()
            scalers = load_scalers()
            
            # --- Preprocess input data ---
            processed_input = preprocess_input(input_df, encoders, scalers)
            
            # --- Predict price ---
            price_pred = catboost_model.predict(processed_input)[0]
            
            # Handle negative predicted price by making it positive
            if price_pred <= 0:
                st.warning("⚠️ Warning: Model predicted a negative price. Converting to positive for analysis.")
                price_pred = abs(price_pred)
                if price_pred == 0:
                    price_pred = 1000000  # Default value if still zero
            
            # --- Calculate valuation using market comparison approach ---
            # Get similar properties from the same city and locality for better comparison
            similar_properties = data[
                (data['City_name'] == city) & 
                (data['Locality_Name'] == locality)
            ].copy()
            
            # If not enough properties in same locality, expand to same city
            if len(similar_properties) < 20:
                similar_properties = data[data['City_name'] == city].copy()
            
            if len(similar_properties) > 0:
                # Prepare features for similar properties
                feature_columns = ['Area','City_name', 'Locality_Name', 'Sub_urban_name', 'Property_type', 'No_of_BHK', 'is_furnished','Property_building_status', 'is_ready_to_move', 'is_PentaHouse' ,'is_Apartment','Property_status']
                similar_features = similar_properties[feature_columns].copy()
                
                # Preprocess similar properties
                processed_similar = preprocess_input(similar_features, encoders, scalers)
                
                # Predict prices for similar properties
                similar_predicted = catboost_model.predict(processed_similar)
                
                # Convert actual prices to numeric
                similar_prices = similar_properties['Price'].astype(str).str.replace(',', '').str.strip()
                similar_actual = pd.to_numeric(similar_prices, errors='coerce')
                
                # Calculate residuals for similar properties
                valid_mask = ~similar_actual.isna()
                if valid_mask.sum() > 10:  # Need sufficient data points
                    similar_residuals = similar_actual[valid_mask] - similar_predicted[valid_mask]
                    
                    # Calculate the residual for the new property
                    # Since we don't have an actual price for the new property, we'll use a different approach
                    # We'll compare the predicted price to the market distribution
                    
                    # Get the distribution of actual prices in the market
                    market_prices = similar_actual[valid_mask]
                    market_predicted = similar_predicted[valid_mask]
                    
                    # Calculate price per sq ft for the new property
                    new_price_per_sqft = price_pred / area
                    
                    # Handle negative price per sq ft by making it positive
                    if new_price_per_sqft <= 0:
                        #st.warning("⚠️ Warning: Calculated negative price per sq ft. Converting to positive for analysis.")
                        new_price_per_sqft = abs(new_price_per_sqft)
                        if new_price_per_sqft == 0:
                            new_price_per_sqft = 10000  # Default value if still zero
                    
                    # Calculate price per sq ft for market properties
                    market_price_per_sqft = market_prices / similar_properties[valid_mask]['Area']
                    
                    # Handle negative market prices per sq ft by making them positive
                    market_price_per_sqft = market_price_per_sqft.apply(lambda x: abs(x) if x <= 0 else x)
                    
                    # Validate market price per sq ft - filter out zero values
                    valid_price_per_sqft = market_price_per_sqft[market_price_per_sqft > 0]
                    
                    if len(valid_price_per_sqft) < 5:
                        #st.warning("⚠️ Insufficient valid market data for comparison. Using fallback method.")
                        label = "Correctly Priced"
                        confidence = "Low"
                        market_median = new_price_per_sqft  # Use predicted as reference
                    else:
                        # Calculate percentiles of the market price per sq ft distribution
                        market_25th = valid_price_per_sqft.quantile(0.25)
                        market_75th = valid_price_per_sqft.quantile(0.75)
                        market_median = valid_price_per_sqft.median()
                    
                    # Classify based on where the new property falls in the market distribution
                    if new_price_per_sqft < market_25th:
                        # Below 25th percentile - likely undervalued
                        label = "Undervalued"
                    elif new_price_per_sqft > market_75th:
                        # Above 75th percentile - likely overvalued
                        label = "Overvalued"
                    else:
                        # Between 25th and 75th percentile - correctly priced
                        label = "Correctly Priced"
                        
                    # Add confidence indicator based on how far from median
                    price_diff_percent = abs(new_price_per_sqft - market_median) / market_median * 100
                    if price_diff_percent > 30:
                        confidence = "High"
                    elif price_diff_percent > 15:
                        confidence = "Medium"
                    else:
                        confidence = "Low"
                        
                else:
                    # Fallback: use a simple approach based on price prediction
                    # Calculate price per sq ft for the new property
                    price_per_sqft = price_pred / area
                    
                    # Get median price per sq ft for similar properties
                    similar_price_per_sqft = similar_actual[valid_mask] / similar_properties[valid_mask]['Area']
                    median_price_per_sqft = similar_price_per_sqft.median()
                    
                    # Simple classification based on price per sq ft comparison
                    if price_per_sqft < median_price_per_sqft * 0.9:
                        label = "Undervalued"
                    elif price_per_sqft > median_price_per_sqft * 1.1:
                        label = "Overvalued"
                    else:
                        label = "Correctly Priced"
                    confidence = "Low"
            else:
                label = "Correctly Priced"  # Default fallback
                confidence = "Low"
            
            st.success(f"Predicted Price: ₹ {price_pred:,.2f}")
            
            # Display valuation with color coding and confidence
            if label == "Undervalued":
                st.success(f"✅ **Valuation:** {label} (Good Deal!)")
            elif label == "Overvalued":
                st.error(f"⚠️ **Valuation:** {label} (Overpriced)")
            else:
                st.info(f"📊 **Valuation:** {label}")
            
            # Show market context
            if 'market_median' in locals():
                st.write(f"📈 **Market Context:** Your property's price per sq ft (₹{new_price_per_sqft:,.0f}) vs Market median (₹{market_median:,.0f})")
            
        except Exception as e:
            st.error(f"Error during prediction: {str(e)}")
            st.info("Please check if all model files are available in the models directory.")
    except Exception as e:
        st.error(f"Error during prediction: {str(e)}")
        st.info("Please check if all model files are available in the models directory.")

# --- Find Similar Properties Functionality ---
elif find_similar_button:
    st.markdown("---")
    st.subheader("🔍 Similar Properties Results")
    
    try:
        # Create a scoring function to find similar properties
        def calculate_similarity_score(row):
            score = 0
            
            # Exact matches get highest scores
            if row['City_name'] == city:
                score += 100
            if row['Sub_urban_name'] == suburb:
                score += 50
            if row['Locality_Name'] == locality:
                score += 30
            if row['Property_type'] == property_type:
                score += 20
            if row['No_of_BHK'] == bhk:
                score += 15
            if row['is_furnished'] == furnished:
                score += 10
            if row['Property_building_status'] == building_status:
                score += 10
            if row['is_ready_to_move'] == ready_to_move:
                score += 5
            if row['is_PentaHouse'] == pentahouse:
                score += 5
            if row['is_Apartment'] == apartment:
                score += 5
            if row['Property_status'] == property_status:
                score += 5
            
            # Size similarity (closer size gets higher score)
            size_diff = abs(row['Area'] - area)
            if size_diff <= 100:
                score += 20
            elif size_diff <= 200:
                score += 15
            elif size_diff <= 500:
                score += 10
            elif size_diff <= 1000:
                score += 5
            
            return score
        
        # Apply similarity scoring to all properties
        data['similarity_score'] = data.apply(calculate_similarity_score, axis=1)
        
        # Filter properties with at least some similarity (score > 0)
        similar_properties = data[data['similarity_score'] > 0].copy()
        
        if len(similar_properties) > 0:
            # Sort by similarity score and get top matches
            similar_properties = similar_properties.sort_values('similarity_score', ascending=False)
            recommended = similar_properties.head(min(50, len(similar_properties)))  # Get top 50 for pagination
            
            # Impute NaN values with most frequent values in that city
            columns_to_impute = ['Sub_urban_name', 'Locality_Name', 'Property_type', 'No_of_BHK', 
                                'is_furnished', 'Property_building_status', 'is_ready_to_move', 
                                'is_PentaHouse', 'is_Apartment', 'Property_status']
            
            for col in columns_to_impute:
                if col in recommended.columns:
                    recommended = impute_nan_with_city_mode(recommended, col)
            
            # Handle NaN Property_Name values
            if 'Property_Name' in recommended.columns:
                recommended['Property_Name'] = recommended['Property_Name'].fillna(recommended['Locality_Name'])
            
            st.success(f"Found {len(recommended)} similar properties based on your preferences!")
            
            # Store the result set for PandasAI queries
            st.session_state.similar_properties_result = recommended.copy()
            
            # Display all properties in a vertically scrollable list
            st.write(f"Showing all {len(recommended)} properties:")
            for idx, property_data in recommended.iterrows():
                # Handle NaN Property_Name
                property_name = property_data['Property_Name'] if pd.notna(property_data['Property_Name']) else property_data['Locality_Name']
                
                with st.expander(f"🏠 {property_name} - ₹{property_data['Price']:,}", expanded=False):
                    # Create a two-column layout for the card
                    col1, col2 = st.columns([2, 1])
                    
                    with col1:
                        st.markdown(f"**Property Details:**")
                        st.write(f"📍 **Location:** {property_data['Locality_Name']}, {property_data['City_name']}")
                        st.write(f"📏 **Size:** {property_data['Size']}")
                        st.write(f"💰 **Price:** ₹{property_data['Price']:,}")
                        
                        # Show matching criteria
                        matches = []
                        if property_data['City_name'] == city:
                            matches.append("✅ City")
                        if property_data['Sub_urban_name'] == suburb:
                            matches.append("✅ Suburb")
                        if property_data['Locality_Name'] == locality:
                            matches.append("✅ Locality")
                        if property_data['Property_type'] == property_type:
                            matches.append("✅ Property Type")
                        if property_data['No_of_BHK'] == bhk:
                            matches.append("✅ BHK")
                        
                        if matches:
                            st.write(f"**Matches:** {', '.join(matches)}")
                    
                    with col2:
                        st.markdown(f"**Quick Actions:**")
                        
                        # Check if URL column exists and create clickable link
                        if 'URL' in recommended.columns:
                            property_url = property_data['URL']
                            if pd.notna(property_url) and property_url:
                                st.markdown(f"[🔗 View Property Details]({property_url})")
                        
                        # Check if Project_URL column exists and create clickable link
                        if 'Project_URL' in recommended.columns:
                            project_url = property_data['Project_URL']
                            if pd.notna(project_url) and project_url:
                                st.markdown(f"[🏗️ View Project Details]({project_url})")
                        
                        # Add description if available
                        if 'Description' in recommended.columns:
                            description = property_data['Description']
                            if pd.notna(description) and description:
                                st.markdown("**Description:**")
                                st.write(description[:200] + "..." if len(str(description)) > 200 else description)
                        
                        # Additional property details if available
                        additional_details = []
                        if 'No_of_BHK' in recommended.columns:
                            bhk_val = property_data['No_of_BHK']
                            if pd.notna(bhk_val):
                                additional_details.append(f"🏠 {bhk_val} BHK")
                        
                        if 'Property_type' in recommended.columns:
                            prop_type = property_data['Property_type']
                            if pd.notna(prop_type):
                                additional_details.append(f"🏢 {prop_type}")
                        
                        if 'is_furnished' in recommended.columns:
                            furnished_val = property_data['is_furnished']
                            if pd.notna(furnished_val):
                                additional_details.append(f"🪑 {furnished_val}")
                        
                        if additional_details:
                            st.markdown("**Additional Info:**")
                            for detail in additional_details:
                                st.write(detail)
                    
                    # Add a separator between cards
                    st.markdown("---")
            st.markdown("---")
            # Show all properties on the map
            st.subheader("Map of All Properties")
            if 'Latitude' in recommended.columns and 'Longitude' in recommended.columns:
                map_data = recommended[['Latitude', 'Longitude']].copy()
                map_data.columns = ['latitude', 'longitude']
                st.map(map_data)
            else:
                st.warning("Latitude/Longitude columns not found in data.")
            
        else:
            st.warning("No properties found matching your criteria. Try relaxing some filters.")
            
    except Exception as e:
        st.error(f"Error finding similar properties: {str(e)}")
        st.info("Please check if all required data is available.")

# --- PandasAI Processing Function ---
def process_pandasai_query(query, df):
    """
    Process PandasAI queries for sorting, filtering, and statistical operations
    """
    try:
        query_lower = query.lower()
        
        # Combined filtering and sorting operations
        if 'sort' in query_lower or 'filter' in query_lower or 'show' in query_lower or 'find' in query_lower:
            # Start with the full dataset
            result = df.copy()
            filters_applied = []
            sort_applied = None
            
            # Extract price filter
            import re
            price_match = re.search(r'(\d+)\s*(lakh|lac|cr|crore|k)', query_lower)
            if price_match:
                amount = float(price_match.group(1))
                unit = price_match.group(2)
                
                if unit in ['lakh', 'lac']:
                    max_price = amount * 100000
                elif unit in ['cr', 'crore']:
                    max_price = amount * 10000000
                elif unit == 'k':
                    max_price = amount * 1000
                else:
                    max_price = amount
                
                result = result[result['Price'] <= max_price]
                filters_applied.append(f"Price ≤ ₹{amount} {unit}")
            
            # Extract city filter
            cities = df['City_name'].unique()
            city_found = False
            for city in cities:
                if city.lower() in query_lower:
                    result = result[result['City_name'] == city]
                    filters_applied.append(f"City = {city}")
                    city_found = True
                    break
            
            # If no exact city match, try partial matching
            if not city_found:
                for city in cities:
                    if any(word in city.lower() for word in search_terms):
                        result = result[result['City_name'] == city]
                        filters_applied.append(f"City = {city}")
                        break
            
            # Extract BHK filter
            bhk_match = re.search(r'(\d+)\s*bhk', query_lower)
            if bhk_match:
                bhk_num = int(bhk_match.group(1))
                result = result[result['No_of_BHK'] == bhk_num]
                filters_applied.append(f"BHK = {bhk_num}")
            
            # Extract furnished filter
            if 'furnished' in query_lower:
                if 'unfurnished' in query_lower:
                    result = result[result['is_furnished'] == 'Unfurnished']
                    filters_applied.append("Furnished = Unfurnished")
                else:
                    result = result[result['is_furnished'] == 'Furnished']
                    filters_applied.append("Furnished = Yes")
            
            # Extract meaningful search terms (exclude common words)
            common_words = {
                'show', 'find', 'filter', 'by', 'in', 'and', 'or', 'properties', 'property', 
                'price', 'city', 'bhk', 'furnished', 'unfurnished', 'lakh', 'lac', 'cr', 'crore', 'k', 
                'sort', 'ascending', 'descending', 'order', 'high', 'low', 'large', 'small', 'the',
                'results', 'locality'
            }
            
            # Extract meaningful search terms
            words = query_lower.split()
            search_terms = []
            for word in words:
                # Clean the word (remove punctuation)
                clean_word = re.sub(r'[^\w\s]', '', word)
                if clean_word and len(clean_word) > 2 and clean_word not in common_words:
                    search_terms.append(clean_word)
            
            if search_terms:
                # Remove city terms from search terms since we already filtered by city
                city_terms = [city.lower() for city in cities]
                location_terms = [term for term in search_terms if term not in city_terms]
                
                if location_terms:
                    # Search for locality/suburb terms
                    matching_rows = []
                    for idx, row in result.iterrows():
                        # Check if ANY location term is found in locality or suburb
                        location_found = False
                        for term in location_terms:
                            # Check in Locality_Name and Sub_urban_name
                            for col in ['Locality_Name', 'Sub_urban_name']:
                                if col in row and pd.notna(row[col]):
                                    cell_value = str(row[col]).lower()
                                    if term in cell_value:
                                        location_found = True
                                        break
                            if location_found:
                                break
                        
                        if location_found:
                            matching_rows.append(idx)
                    
                    if matching_rows:
                        result = result.loc[matching_rows]
                        filters_applied.append(f"Locality: {' OR '.join(location_terms)}")
                    else:
                        # No locality matches found
                        result = result.iloc[0:0]  # Empty DataFrame
                        filters_applied.append(f"No properties found in locality: {' OR '.join(location_terms)}")
            
            # Apply sorting after filtering
            if 'sort' in query_lower:
                if 'price' in query_lower:
                    if 'ascending' in query_lower or 'low to high' in query_lower:
                        result = result.sort_values('Price', ascending=True)
                        sort_applied = "Price (Low to High)"
                    elif 'descending' in query_lower or 'high to low' in query_lower:
                        result = result.sort_values('Price', ascending=False)
                        sort_applied = "Price (High to Low)"
                    else:
                        # Default to descending for price sorting
                        result = result.sort_values('Price', ascending=False)
                        sort_applied = "Price (High to Low)"
                elif 'area' in query_lower or 'size' in query_lower:
                    if 'ascending' in query_lower or 'small to large' in query_lower:
                        result = result.sort_values('Area', ascending=True)
                        sort_applied = "Area (Small to Large)"
                    elif 'descending' in query_lower or 'large to small' in query_lower:
                        result = result.sort_values('Area', ascending=False)
                        sort_applied = "Area (Large to Small)"
                    else:
                        result = result.sort_values('Area', ascending=False)
                        sort_applied = "Area (Large to Small)"
                elif 'bhk' in query_lower:
                    result = result.sort_values('No_of_BHK', ascending=False)
                    sort_applied = "BHK (High to Low)"
            
            # Ensure result is a DataFrame and not empty before sorting
            if len(result) > 0 and not result.empty:
                # Reset index after filtering and sorting
                result = result.reset_index(drop=True)
            
            # Create clean result message
            if len(result) > 0:
                if sort_applied:
                    return result, f"Found {len(result)} properties - Sorted by {sort_applied}"
                else:
                    return result, f"Found {len(result)} properties"
            else:
                return result, f"No properties found matching your criteria"
        

        
        # Statistical queries
        elif 'stat' in query_lower or 'average' in query_lower or 'mean' in query_lower or 'median' in query_lower:
            if 'price' in query_lower:
                avg_price = df['Price'].mean()
                median_price = df['Price'].median()
                min_price = df['Price'].min()
                max_price = df['Price'].max()
                
                stats_df = pd.DataFrame({
                    'Metric': ['Average Price', 'Median Price', 'Minimum Price', 'Maximum Price'],
                    'Value (₹)': [f"{avg_price:,.0f}", f"{median_price:,.0f}", f"{min_price:,.0f}", f"{max_price:,.0f}"]
                })
                return stats_df, "Price Statistics"
            
            elif 'area' in query_lower or 'size' in query_lower:
                avg_area = df['Area'].mean()
                median_area = df['Area'].median()
                min_area = df['Area'].min()
                max_area = df['Area'].max()
                
                stats_df = pd.DataFrame({
                    'Metric': ['Average Area', 'Median Area', 'Minimum Area', 'Maximum Area'],
                    'Value (sq ft)': [f"{avg_area:,.0f}", f"{median_area:,.0f}", f"{min_area:,.0f}", f"{max_area:,.0f}"]
                })
                return stats_df, "Area Statistics"
            
            elif 'bhk' in query_lower:
                bhk_counts = df['No_of_BHK'].value_counts().sort_index()
                stats_df = pd.DataFrame({
                    'BHK': bhk_counts.index,
                    'Count': bhk_counts.values
                })
                return stats_df, "BHK Distribution"
        
        # Count queries
        elif 'count' in query_lower or 'how many' in query_lower:
            if 'total' in query_lower:
                return f"Total properties: {len(df)}", "Property Count"
            elif 'city' in query_lower:
                city_counts = df['City_name'].value_counts()
                stats_df = pd.DataFrame({
                    'City': city_counts.index,
                    'Property Count': city_counts.values
                })
                return stats_df, "Properties by City"
        
        # Default response for unrecognized queries
        else:
            return df.head(10), f"Showing first 10 properties"
    
    except Exception as e:
        return f"Error processing query: {str(e)}", "Error"
    
    # Fallback return in case function reaches here
    return df.head(10), f"Showing first 10 properties"



# --- PandasAI Results Display ---
if hasattr(st.session_state, 'pandasai_query') and st.session_state.pandasai_query:
    st.markdown("---")
    st.subheader("📊 Query Results")
    
    # Determine which dataset to use (result set or original)
    if hasattr(st.session_state, 'similar_properties_result') and st.session_state.similar_properties_result is not None:
        # Use the result set from similar properties search
        query_df = st.session_state.similar_properties_result
        st.info("🔍 Using Similar Properties Result Set")
    else:
        # Use the original dataset
        query_df = st.session_state.pandasai_data
        st.info("📋 Using Original Dataset")
    
    # Process the query
    result, title = process_pandasai_query(st.session_state.pandasai_query, query_df)
    
    # Display the query
    st.write(f"**Query:** {st.session_state.pandasai_query}")
    st.write(f"**Result:** {title}")
    
    # Display results based on type
    if isinstance(result, pd.DataFrame):
        if len(result) > 0:
            # Show summary statistics
            st.write(f"**Found {len(result)} results**")
            
            # Check if this is property data (has required columns)
            required_columns = ['City_name', 'Locality_Name', 'Price', 'Size']
            if all(col in result.columns for col in required_columns):
                # Display properties in expandable cards format
                st.write(f"Showing all {len(result)} properties:")
                for idx, property_data in result.iterrows():
                    # Handle NaN Property_Name
                    if 'Property_Name' in result.columns and pd.notna(property_data['Property_Name']):
                        property_name = property_data['Property_Name']
                    else:
                        property_name = property_data['Locality_Name']
                    
                    with st.expander(f"🏠 {property_name} - ₹{property_data['Price']:,}", expanded=False):
                        # Create a two-column layout for the card
                        col1, col2 = st.columns([2, 1])
                        
                        with col1:
                            st.markdown(f"**Property Details:**")
                            st.write(f"📍 **Location:** {property_data['Locality_Name']}, {property_data['City_name']}")
                            st.write(f"📏 **Size:** {property_data['Size']}")
                            st.write(f"💰 **Price:** ₹{property_data['Price']:,}")
                            
                            # Show additional details if available
                            additional_details = []
                            if 'No_of_BHK' in result.columns and pd.notna(property_data['No_of_BHK']):
                                additional_details.append(f"🏠 {property_data['No_of_BHK']} BHK")
                            if 'Property_type' in result.columns and pd.notna(property_data['Property_type']):
                                additional_details.append(f"🏢 {property_data['Property_type']}")
                            if 'is_furnished' in result.columns and pd.notna(property_data['is_furnished']):
                                additional_details.append(f"🪑 {property_data['is_furnished']}")
                            if 'Sub_urban_name' in result.columns and pd.notna(property_data['Sub_urban_name']):
                                additional_details.append(f"🏘️ {property_data['Sub_urban_name']}")
                            
                            if additional_details:
                                st.write(f"**Additional Info:** {', '.join(additional_details)}")
                        
                        with col2:
                            st.markdown(f"**Quick Actions:**")
                            
                            # Check if URL column exists and create clickable link
                            if 'URL' in result.columns and pd.notna(property_data['URL']) and property_data['URL']:
                                st.markdown(f"[🔗 View Property Details]({property_data['URL']})")
                            
                            # Check if Project_URL column exists and create clickable link
                            if 'Project_URL' in result.columns and pd.notna(property_data['Project_URL']) and property_data['Project_URL']:
                                st.markdown(f"[🏗️ View Project Details]({property_data['Project_URL']})")
                            
                            # Add description if available
                            if 'Description' in result.columns and pd.notna(property_data['Description']) and property_data['Description']:
                                st.markdown("**Description:**")
                                st.write(property_data['Description'][:200] + "..." if len(str(property_data['Description'])) > 200 else property_data['Description'])
                        
                        # Add a separator between cards
                        st.markdown("---")
            else:
                # Display non-property data as regular dataframe
                if len(result) <= 50:
                    st.dataframe(result, use_container_width=True)
                else:
                    st.write("**Showing first 50 results:**")
                    st.dataframe(result.head(50), use_container_width=True)
                    st.write(f"*... and {len(result) - 50} more results*")
        else:
            st.warning("No results found for your query.")
    else:
        # Display text result
        st.write(result)
    
    # Clear query button
    if st.button("🗑️ Clear Query Results"):
        del st.session_state.pandasai_query
        st.rerun()

# --- Main App Layout ---
else:
    st.markdown("---")
    st.subheader("🏠 Property Price Prediction & Analysis")
    st.write("Welcome to the Property Price Prediction and Analysis tool. Use the buttons above to predict property prices or find similar properties.")