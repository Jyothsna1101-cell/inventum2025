import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

# Load dataset
df = pd.read_csv("2020.csv")

# Basic Info
print(df.info())

# Summary Statistics
print(df.describe())

# Check for missing values
print(df.isnull().sum())

# Distribution of Happiness Scores
plt.figure(figsize=(8, 5))
sns.histplot(df["Happiness score"], bins=20, kde=True)
plt.title("Distribution of Happiness Scores")
plt.show()

# Correlation Heatmap
plt.figure(figsize=(10, 6))
sns.heatmap(df.corr(), annot=True, cmap="coolwarm", fmt=".2f")
plt.title("Correlation Heatmap")
plt.show()
