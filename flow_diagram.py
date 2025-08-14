import matplotlib.pyplot as plt
import matplotlib.patches as patches
from matplotlib.patches import FancyBboxPatch, ConnectionPatch
import numpy as np

# Set up the figure and axis
fig, ax = plt.subplots(1, 1, figsize=(16, 12))
ax.set_xlim(0, 10)
ax.set_ylim(0, 12)
ax.axis('off')

# Define colors
colors = {
    'data': '#E8F4FD',
    'etl': '#FFE6CC',
    'ml': '#E6F3FF',
    'ui': '#F0E6FF',
    'output': '#E6FFE6'
}

# Function to create rounded rectangle boxes
def create_box(x, y, width, height, text, color, fontsize=10):
    box = FancyBboxPatch((x, y), width, height,
                         boxstyle="round,pad=0.1",
                         facecolor=color,
                         edgecolor='black',
                         linewidth=1.5)
    ax.add_patch(box)
    ax.text(x + width/2, y + height/2, text, 
            ha='center', va='center', fontsize=fontsize, fontweight='bold')

# Function to create arrows
def create_arrow(start_x, start_y, end_x, end_y, color='black'):
    arrow = ConnectionPatch((start_x, start_y), (end_x, end_y), 
                          "data", "data",
                          arrowstyle="->", shrinkA=5, shrinkB=5,
                          mutation_scale=20, fc=color, ec=color, linewidth=2)
    ax.add_patch(arrow)

# Title
ax.text(5, 11.5, 'Housing Price Prediction & Analysis - Complete Pipeline', 
        ha='center', va='center', fontsize=16, fontweight='bold')

# Phase 1: Data Ingestion
create_box(0.5, 10, 2, 0.8, 'Raw Data\nMakaan_Properties_Buy.csv\n(172MB)', colors['data'], 9)

# Phase 2: ETL Process
create_box(3.5, 10, 2, 0.8, 'ETL Pipeline\nData Cleaning & Transformation', colors['etl'], 9)
create_arrow(2.5, 10.4, 3.5, 10.4)

# ETL Sub-processes
create_box(0.5, 8.5, 1.8, 0.6, 'Size/Price\nNormalization', colors['etl'], 8)
create_box(2.5, 8.5, 1.8, 0.6, 'Date\nTransformation', colors['etl'], 8)
create_box(4.5, 8.5, 1.8, 0.6, 'Missing Value\nImputation', colors['etl'], 8)
create_box(6.5, 8.5, 1.8, 0.6, 'Feature\nEngineering', colors['etl'], 8)

# Arrows from ETL to sub-processes
create_arrow(4.5, 9.2, 1.4, 9.1)
create_arrow(4.5, 9.2, 3.4, 9.1)
create_arrow(4.5, 9.2, 5.4, 9.1)
create_arrow(4.5, 9.2, 7.4, 9.1)

# Phase 3: Feature Engineering
create_box(0.5, 7.5, 2, 0.6, 'Amenities\nExtraction\n(25+ features)', colors['etl'], 8)
create_box(2.8, 7.5, 2, 0.6, 'Label\nEncoding', colors['etl'], 8)
create_box(5.1, 7.5, 2, 0.6, 'Standard\nScaling', colors['etl'], 8)
create_box(7.4, 7.5, 2, 0.6, 'Boolean\nEncoding', colors['etl'], 8)

# Phase 4: Model Training
create_box(2, 6.5, 2, 0.8, 'CatBoost\nRegression\n(Price Prediction)', colors['ml'], 9)
create_box(6, 6.5, 2, 0.8, 'K-means\nClustering\n(Valuation)', colors['ml'], 9)

# Arrows to models
create_arrow(1.4, 7.5, 2, 6.9)
create_arrow(3.4, 7.5, 3, 6.9)
create_arrow(5.4, 7.5, 4, 6.9)
create_arrow(7.4, 7.5, 6, 6.9)

# Phase 5: Model Evaluation
create_box(1, 5.5, 2, 0.6, 'R² Score\nEvaluation', colors['ml'], 8)
create_box(3.5, 5.5, 2, 0.6, 'Silhouette\nScore', colors['ml'], 8)
create_box(6, 5.5, 2, 0.6, 'Feature\nImportance', colors['ml'], 8)

# Arrows from models to evaluation
create_arrow(3, 6.5, 2, 6.1)
create_arrow(3, 6.5, 4.5, 6.1)
create_arrow(7, 6.5, 7, 6.1)

# Phase 6: Model Persistence
create_box(2, 4.5, 2, 0.6, 'cat_boost.pkl\n(3.9MB)', colors['output'], 8)
create_box(4.5, 4.5, 2, 0.6, 'Clustering.pkl\n(260KB)', colors['output'], 8)
create_box(7, 4.5, 2, 0.6, 'encoders.pkl\nscalers.pkl', colors['output'], 8)

# Arrows to model files
create_arrow(2, 5.5, 3, 5.1)
create_arrow(4.5, 5.5, 5.5, 5.1)
create_arrow(7, 5.5, 8, 5.1)

# Phase 7: Web Application
create_box(1, 3.5, 2, 0.8, 'Streamlit\nWeb App', colors['ui'], 9)
create_box(4, 3.5, 2, 0.8, 'Model\nLoading', colors['ui'], 9)
create_box(7, 3.5, 2, 0.8, 'User\nInterface', colors['ui'], 9)

# Arrows to web app
create_arrow(3, 4.5, 2, 4.3)
create_arrow(5.5, 4.5, 5, 4.3)
create_arrow(8, 4.5, 8, 4.3)

# Phase 8: User Features
create_box(0.5, 2.5, 2, 0.6, 'Price\nPrediction', colors['ui'], 8)
create_box(2.8, 2.5, 2, 0.6, 'Similar\nProperties', colors['ui'], 8)
create_box(5.1, 2.5, 2, 0.6, 'Valuation\nAnalysis', colors['ui'], 8)
create_box(7.4, 2.5, 2, 0.6, 'Natural Language\nQueries', colors['ui'], 8)

# Arrows to features
create_arrow(2, 3.5, 1.5, 3.1)
create_arrow(5, 3.5, 3.8, 3.1)
create_arrow(8, 3.5, 6.1, 3.1)
create_arrow(8, 3.5, 8.4, 3.1)

# Phase 9: Output
create_box(3.5, 1.5, 3, 0.8, 'Interactive Results\nCharts & Analysis', colors['output'], 9)

# Arrows to output
create_arrow(1.5, 2.5, 3.5, 2.3)
create_arrow(3.8, 2.5, 4, 2.3)
create_arrow(6.1, 2.5, 4.5, 2.3)
create_arrow(8.4, 2.5, 5, 2.3)

# Add phase labels
ax.text(1.5, 9.8, 'Phase 1: Data Ingestion', ha='center', va='center', 
        fontsize=10, fontweight='bold', color='blue')
ax.text(4.5, 9.8, 'Phase 2: ETL Processing', ha='center', va='center', 
        fontsize=10, fontweight='bold', color='blue')
ax.text(4, 8.2, 'Phase 3: Feature Engineering', ha='center', va='center', 
        fontsize=10, fontweight='bold', color='blue')
ax.text(4, 7.2, 'Phase 4: Model Training', ha='center', va='center', 
        fontsize=10, fontweight='bold', color='blue')
ax.text(4, 6.2, 'Phase 5: Model Evaluation', ha='center', va='center', 
        fontsize=10, fontweight='bold', color='blue')
ax.text(4, 5.2, 'Phase 6: Model Persistence', ha='center', va='center', 
        fontsize=10, fontweight='bold', color='blue')
ax.text(4, 4.2, 'Phase 7: Web Application', ha='center', va='center', 
        fontsize=10, fontweight='bold', color='blue')
ax.text(4, 3.2, 'Phase 8: User Features', ha='center', va='center', 
        fontsize=10, fontweight='bold', color='blue')
ax.text(5, 2.2, 'Phase 9: Results & Analysis', ha='center', va='center', 
        fontsize=10, fontweight='bold', color='blue')

# Add legend
legend_elements = [
    patches.Patch(color=colors['data'], label='Data Sources'),
    patches.Patch(color=colors['etl'], label='ETL & Processing'),
    patches.Patch(color=colors['ml'], label='Machine Learning'),
    patches.Patch(color=colors['ui'], label='User Interface'),
    patches.Patch(color=colors['output'], label='Output & Results')
]

ax.legend(handles=legend_elements, loc='upper right', bbox_to_anchor=(0.98, 0.98))

plt.tight_layout()
plt.savefig('housing_price_prediction_flow_diagram.png', dpi=300, bbox_inches='tight')
plt.show()

print("Flow diagram saved as 'housing_price_prediction_flow_diagram.png'")
