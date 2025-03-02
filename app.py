import pandas as pd
import matplotlib.pyplot as plt

# Load dataset
df = pd.read_csv("C:\Users\DVS. BHAVANA\Downloads\2020.csv")

# Basic Info
print(df.info())

# Summary Statistics
print(df.describe())

# Check for missing values
print("Missing Values:\n", df.isnull().sum())

# Distribution of Happiness Scores
plt.figure(figsize=(8, 5))
plt.hist(df["Happiness score"], bins=20, edgecolor="black", alpha=0.7)
plt.title("Distribution of Happiness Scores")
plt.xlabel("Happiness Score")
plt.ylabel("Frequency")
plt.grid(axis="y", linestyle="--", alpha=0.7)
plt.show()

# Correlation Heatmap (without seaborn)
import numpy as np

correlation_matrix = df.corr()

fig, ax = plt.subplots(figsize=(10, 6))
cax = ax.matshow(correlation_matrix, cmap="coolwarm")
plt.colorbar(cax)

ax.set_xticks(np.arange(len(correlation_matrix.columns)))
ax.set_yticks(np.arange(len(correlation_matrix.columns)))

ax.set_xticklabels(correlation_matrix.columns, rotation=90)
ax.set_yticklabels(correlation_matrix.columns)

plt.title("Correlation Heatmap")
plt.show()
