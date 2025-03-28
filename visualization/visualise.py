import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.decomposition import PCA
from mpl_toolkits.mplot3d import Axes3D

# Define file paths
RAW_DATA_PATH = r"data\raw\snRNA_seq_data.csv"  # Update path if necessary
PROCESSED_DATA_PATH = r"data/processed/normalized_data.csv"  # Update path if necessary

# Define visualization directories (assuming folders are already created)
BEFORE_DIR = "visualization/before/"
AFTER_DIR = "visualization/after/"

# Load datasets
df_raw = pd.read_csv(RAW_DATA_PATH).dropna()
df_processed = pd.read_csv(PROCESSED_DATA_PATH).dropna()

# Select only numeric columns for visualization
df_raw_numeric = df_raw.select_dtypes(include=[np.number])
df_processed_numeric = df_processed.select_dtypes(include=[np.number])

# Function to save histogram
def save_histogram(df, folder, title):
    for col in df.columns[:5]:  # Visualize only first 5 numeric columns
        plt.figure(figsize=(8, 6))
        plt.hist(df[col], bins=30, edgecolor="black", alpha=0.7)
        plt.xlabel("Expression Level")
        plt.ylabel("Frequency")
        plt.title(f"Histogram of {col} - {title}")
        plt.savefig(f"{folder}histogram_{col}.png", dpi=300)
        plt.close()

# Function to save boxplot
def save_boxplot(df, folder, title):
    for col in df.columns[:5]:
        plt.figure(figsize=(8, 6))
        sns.boxplot(data=df[col])
        plt.title(f"Boxplot of {col} - {title}")
        plt.savefig(f"{folder}boxplot_{col}.png", dpi=300)
        plt.close()

# Function to save scatter plot of first two columns
def save_scatter(df, folder, title):
    if df.shape[1] < 2:
        print(f"Not enough columns for scatter plot in {title}")
        return

    plt.figure(figsize=(8, 6))
    plt.scatter(df.iloc[:, 0], df.iloc[:, 1], alpha=0.5)
    plt.xlabel(df.columns[0])
    plt.ylabel(df.columns[1])
    plt.title(f"Scatter Plot: {df.columns[0]} vs {df.columns[1]} - {title}")
    plt.savefig(f"{folder}scatter_{df.columns[0]}_vs_{df.columns[1]}.png", dpi=300)
    plt.close()

# Function to save PCA plots
def save_pca(df, folder, title):
    X = df.select_dtypes(include=[np.number])

    # 2D PCA
    pca_2d = PCA(n_components=2)
    X_pca_2d = pca_2d.fit_transform(X)

    plt.figure(figsize=(8, 6))
    plt.scatter(X_pca_2d[:, 0], X_pca_2d[:, 1], alpha=0.6)
    plt.xlabel("PCA Component 1")
    plt.ylabel("PCA Component 2")
    plt.title(f"{title} - 2D PCA")
    plt.savefig(f"{folder}pca_2d.png", dpi=300)
    plt.close()

    # 3D PCA
    pca_3d = PCA(n_components=3)
    X_pca_3d = pca_3d.fit_transform(X)

    fig = plt.figure(figsize=(8, 6))
    ax = fig.add_subplot(111, projection="3d")
    ax.scatter(X_pca_3d[:, 0], X_pca_3d[:, 1], X_pca_3d[:, 2], alpha=0.6)
    ax.set_xlabel("PCA Component 1")
    ax.set_ylabel("PCA Component 2")
    ax.set_zlabel("PCA Component 3")
    ax.set_title(f"{title} - 3D PCA")
    plt.savefig(f"{folder}pca_3d.png", dpi=300)
    plt.close()

# Function to save heatmap
def save_heatmap(df, folder, title):
    plt.figure(figsize=(12, 8))
    sns.heatmap(df.corr(), cmap="coolwarm", annot=False)
    plt.title(f"Correlation Heatmap - {title}")
    plt.savefig(f"{folder}heatmap.png", dpi=300)
    plt.close()

# Generate and save visualizations
print("Saving visualizations for raw data (before preprocessing)...")
save_histogram(df_raw_numeric, BEFORE_DIR, "Raw Data")
save_boxplot(df_raw_numeric, BEFORE_DIR, "Raw Data")
save_scatter(df_raw_numeric, BEFORE_DIR, "Raw Data")
save_pca(df_raw_numeric, BEFORE_DIR, "Raw Data")
save_heatmap(df_raw_numeric, BEFORE_DIR, "Raw Data")

print("Saving visualizations for processed data (after preprocessing)...")
save_histogram(df_processed_numeric, AFTER_DIR, "Processed Data")
save_boxplot(df_processed_numeric, AFTER_DIR, "Processed Data")
save_scatter(df_processed_numeric, AFTER_DIR, "Processed Data")
save_pca(df_processed_numeric, AFTER_DIR, "Processed Data")
save_heatmap(df_processed_numeric, AFTER_DIR, "Processed Data")

print("All visualizations saved!")
