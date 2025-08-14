from datetime import timedelta
import pandas as pd
import numpy as np
from catboost import CatBoostRegressor
from sklearn.cluster import KMeans
from sklearn.preprocessing import LabelEncoder, StandardScaler, OneHotEncoder
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split, learning_curve, GridSearchCV
from sklearn.metrics import r2_score, adjusted_rand_score, silhouette_score
import pickle

x = df[['Area','City_name', 'Locality_Name', 'Sub_urban_name', 'Property_type', 'No_of_BHK', 'is_furnished','Property_building_status', 'is_ready_to_move', 'is_PentaHouse' ,'is_Apartment','Property_status']]
y = df['Price']

x_train, x_test, y_train, y_test = train_test_split(x, y, random_state=42, test_size=0.2)

# Catboost model
categorical_features = x.select_dtypes(include='object').columns.tolist()

model = CatBoostRegressor(
    iterations=6000,
    learning_rate=0.1,
    depth=6,
    eval_metric='R2',
    cat_features=categorical_features,
    verbose=100,
    random_state=42
)
model.fit(x_train, y_train, eval_set=(x_test, y_test), early_stopping_rounds=200)

y_pred = model.predict(x_test)

print("R2 Score : ", r2_score(y_test, y_pred))
pickle.dump(model,open('cat_boost.pkl','wb'))

importances = model.feature_importances_
features = x_train.columns

# Create DataFrame
feat_imp = pd.DataFrame({'Feature': features, 'Importance': importances})
feat_imp = feat_imp.sort_values(by='Importance', ascending=True)

# Plot
plt.figure(figsize=(10, 12))
plt.barh(feat_imp['Feature'], feat_imp['Importance'], color='skyblue')
plt.title('Feature Importance (CatBoost)', fontsize=14)
plt.xlabel('Importance Score')
plt.tight_layout()
plt.show()

#------------------------------------------ Clustering
residual = y_test - y_pred

cluster = pd.DataFrame(x_test, columns=x.columns)
cluster['Residual'] = residual.values
cluster['Predicted_Price'] = y_pred
cluster['Actual_Price'] = y_test.values

kmeans = KMeans(n_clusters=3, random_state=42)
cluster['cluster'] = kmeans.fit_predict(cluster[['Residual', 'Predicted_Price']])
pickle.dump(kmeans,open('Clustering.pkl','wb'))


# Compute average residual for each cluster
mean_residuals = cluster.groupby('cluster')['Residual'].mean().sort_values()

# Map cluster IDs to labels based on sorted residuals
label_map = {
    mean_residuals.index[0]: 'Undervalued',
    mean_residuals.index[1]: 'Correctly Priced',
    mean_residuals.index[2]: 'Overvalued'
}

# Apply mapping to cluster
cluster['Value_Status'] = cluster['cluster'].map(label_map)
df = df.loc[cluster.index]
df['Status'] = cluster['Value_Status'].values

sns.scatterplot(data=cluster, x='Predicted_Price', y='Residual', hue='Value_Status', palette={'Undervalued': 'green', 'Correctly Priced': 'blue', 'Overvalued': 'red'}, s=80, edgecolor='black')
plt.axhline(0, color='gray', linestyle='--')
plt.title("Residual vs Predicted Price")
plt.xlabel("Predicted Price")
plt.ylabel("Residual")
plt.legend(title='Valuation')
plt.grid(True)
plt.show()

sil_score = silhouette_score(cluster[['Residual', 'Predicted_Price']], cluster['cluster'])
print(f"Silhouette Score: {sil_score:.3f}")
