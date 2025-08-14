import pandas as pd
import matplotlib.pyplot as plt

# Load CSV file
df = pd.read_csv('D:\\Housing_Project_\\Housing_CatBoost_Kmeans\\Streamlit_UI\\data\\training_log.csv')

# Convert scores to percentages
df['learn_pct'] = df['learn'] * 100
df['test_pct'] = df['test'] * 100
df['best_pct'] = df['best'] * 100

# Plot
plt.figure(figsize=(10,6))
plt.plot(df['iteration'], df['learn_pct'], label='Training Score', color='teal')
plt.plot(df['iteration'], df['test_pct'], label='Validation Score', color='orange')
plt.plot(df['iteration'], df['best_pct'], label='Best Validation', color='green', linestyle='dashed')

plt.xlabel('Iteration')
plt.ylabel('Score (%)')
plt.title('Training vs Validation vs Best Score (%)')
plt.legend()
plt.grid(True)

# Format y-axis as percentage
plt.yticks(range(0, 101, 10), [f"{y}%" for y in range(0, 101, 10)])

plt.tight_layout()
plt.show()
