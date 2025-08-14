# Housing Price Predictor

## About the Project

Housing Price Predictor is an end-to-end real estate analytics solution designed to automate the process of price estimation, region classification, property valuation, and recommendation for residential properties. Leveraging advanced data engineering, machine learning, and interactive visualization tools, this project enables accurate property valuation and personalized recommendations in a scalable, user-friendly workflow.

The system covers:

- Data extraction, transformation, and exploratory analysis on real-estate datasets.
- Geographic region categorization using latitude/longitude data.
- Area-wise price prediction using machine learning (XGBoost).
- Area and price-based clustering with K-Means and custom valuation (overvalued/undervalued labels).
- Data storage in MySQL, visualization with Tableau, and an interactive property recommendation web app via Streamlit.
- Model and data management using AWS S3.
- Optional: Natural language conversational analytics using pandasai for ad-hoc data insights.


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

