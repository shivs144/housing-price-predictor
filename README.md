# Introduction

Accurately estimating real estate prices is crucial for buyers, sellers, and investors. This project provides a robust, data-driven solution for predicting residential property prices, classifying market value, and recommending properties using advanced machine learning techniques. With a user-friendly Streamlit interface and integrations for geospatial and intelligent data analysis, it empowers users to make informed decisions in the real estate market.

# Housing Price Predictor

## Overview

This project predicts residential property prices and provides valuation and recommendations using machine learning. The solution is built with Python, Streamlit, CatBoost, and KMeans clustering.

## Main Features

- **Price Prediction:** Uses CatBoost regression to estimate property prices.
- **Clustering:** KMeans groups properties for recommendations and market segmentation.
- **Valuation:** Flags properties as undervalued, overvalued, or correctly valued based on predicted vs. actual price.
- **Recommendation System:** Suggests properties based on user input and cluster analysis.
- **Geospatial Analysis:** Integrates Geopy for location-based features.
- **Intelligent Data Analysis:** Uses Pandas AI for enhanced dataframe operations.

## Model Metrics

- **R² Score:** Measures regression model accuracy (typically >0.85).
- **Silhouette Score:** Assesses clustering quality (typically >0.6).

## How Valuation Works

- **Undervalued:** Predicted price > actual price by a set threshold.
- **Overvalued:** Predicted price < actual price by a set threshold.
- **Correctly Valued:** Predicted price ≈ actual price.

## Directory Structure

```
data/           # Raw and processed datasets
etl/            # Data preprocessing scripts
modeling/       # Model training and saving
models/         # Saved model files (cat_boost.pkl, Clustering.pkl, encoders.pkl, scalers.pkl)
ui/             # Streamlit app (app.py)
utils/          # Utility functions
requirements.txt
README.md
```

## Installation

1. Clone the repository.
2. Create and activate a Python virtual environment.
3. Install dependencies:
     ```
     pip install -r requirements.txt
     ```

## Usage

- **Run the Streamlit UI:**
    ```
    cd ui
    streamlit run app.py
    ```
- **Train models:**
    ```
    python modeling/save_models.py
    ```

## Integrations

- **CatBoost:** Main regression model for price prediction.
- **KMeans:** Clustering for recommendations.
- **Geopy:** Location-based feature engineering.
- **Pandas AI:** Advanced dataframe analytics.

## Contributing

Pull requests and suggestions are welcome!

## License

MIT License

## Tech Stack

| Component | Technology/Library |
| :-- | :-- |
| **Data Processing** | PySpark |
| **Data Storage** | MySQL |
| **Data Visualization** | Tableau Desktop / Tableau Public |
| **Machine Learning** | XGBoost, scikit-learn (KMeans), pandas |
| **Model Serialization** | pickle |
| **Cloud Storage** | AWS S3, boto3 library |
| **Web App / Interface** | Streamlit |
| **NLP/Sentence Similarity** | sentence-transformers, scikit-learn |
| **Conversational Analytics** | pandasai (optional) |
| **Connector Libraries** | mysql-connector, com.mysql.cj.jdbc.Driver |
| **Utilities** | pandas, numpy, matplotlib, seaborn, etc. |

## Key Features

- **Automated ETL Pipeline:** Clean, transform, and prepare real-estate data using PySpark for robust analysis and modeling.
- **Geographic Enrichment:** Categorizes properties into regions based on their latitude/longitude.
- **AI/ML Modeling:** Predicts prices and classifies properties into market categories using state-of-the-art regression and clustering.
- **Property Valuation:** Flags each property as undervalued, fair, or overvalued with data-driven reasoning.
- **Scalable Data Storage:** Centralized MySQL database for easy data and analytics access.
- **Rich Visual Reports:** Tableau dashboards offer drill-downs for pricing trends, geographical insights, and market segments.
- **Interactive User Interface:** Streamlit app for real-time property recommendation based on advanced search (region, price, BHK, custom text description).
- **Cloud-first Model Serving:** Machine learning models and encoders are stored securely in AWS S3 and loaded on demand.
- **Conversational Insights (Optional):** Natural language querying and visualizations for ad-hoc analytics, powered by pandasai.

## Integrations

- **Pandas AI:**
    Used for enhanced data analysis and intelligent dataframe operations, enabling dynamic and context-aware data transformations within ETL and modeling pipelines.

- **Geopy:**
    Used for geospatial analysis, such as calculating distances and extracting location-based features, improving the accuracy of property price predictions.

## Model Metrics

- **R² Score:**
    Regression models are evaluated using the R² Score, quantifying the proportion of variance in property prices explained by the model. Typical R² scores achieved are above 0.85, indicating strong predictive performance.

- **Silhouette Score:**
    Clustering models are assessed using the Silhouette Score, measuring how well each property fits within its assigned cluster. Scores above 0.6 demonstrate meaningful groupings of properties based on features.

## Recommendation System

The recommendation system suggests properties to users based on their input preferences and clustering results. It works by:

1. Assigning each property to a cluster using unsupervised learning (e.g., KMeans).
2. Matching user preferences to the most relevant cluster.
3. Recommending properties from that cluster with the highest predicted value and best fit.

## Valuation Categories

Properties are classified into three valuation categories:

- **Undervalued:**
    Predicted price is significantly higher than the listed price, indicating a potential bargain.

- **Overvalued:**
    Predicted price is significantly lower than the listed price, suggesting the property may be overpriced.

- **Correctly Valued:**
    Predicted price is close to the listed price, indicating fair market value.

The classification is based on the percentage difference between predicted and actual prices, with customizable thresholds.

## Project Structure Overview

- **ETL \& Data Engineering:** Scripts and notebooks for PySpark preprocessing and EDA.
- **Database Integration:** MySQL DDL/DML for storing the processed data.
- **ML Models:** Training and pickling of predictive/classification/clustering models.
- **Dashboarding:** Tableau workbooks configured to consume data from MySQL.
- **Web App:** Streamlit app for user search and recommendation, with S3 model retrieval.
- **Docs \& Utilities:** README, configuration files, and utility scripts for easy deployment.

## Getting Started

Refer to the detailed documentation and comments within scripts for setup, configuration, and run instructions. Key steps involve:

- Setting up your AWS, MySQL, and Tableau environments.
- Running the PySpark scripts to clean and process data.
- Training and pickling ML models, then uploading to S3.
- Launching the Streamlit interface and connecting all services via provided credentials.

## Project Goals

- Deliver reliable real-estate price prediction and valuation in diverse markets and regions.
- Make advanced analytics accessible to both technical and non-technical users.
- Ensure modularity, scalability, and ease of extension for future enhancements.

For further project details, step-by-step process descriptions, and sample code, please refer to the project’s dedicated documentation and workflow outlined previously.




- Deliver reliable real-estate price prediction and valuation in diverse markets and regions.
- Make advanced analytics accessible to both technical and non-technical users.
- Ensure modularity, scalability, and ease of extension for future enhancements.

For further project details, step-by-step process descriptions, and sample code, please refer to the project’s dedicated documentation and workflow outlined previously.

