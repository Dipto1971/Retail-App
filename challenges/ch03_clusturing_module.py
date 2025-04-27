import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.mixture import GaussianMixture
from sklearn.manifold import TSNE
from sklearn.metrics import silhouette_score, davies_bouldin_score
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler

def run_inventory_clustering(data):
    st.header("Challenge 3: Clustering for Inventory Insights")
    st.write("This section uses clustering to identify inventory patterns.")

    # Define numerical and categorical features for clustering
    numerical_cols = [
        'Inventory Level', 'Units Sold', 'Units Ordered', 'Demand Forecast',
        'Price', 'Discount', 'Competitor Pricing', 'Holiday/Promotion'
    ]
    categorical_cols = ['Category', 'Region', 'Weather Condition', 'Seasonality']

    # Filter features present in the dataset
    numerical_cols = [col for col in numerical_cols if col in data.columns]
    categorical_cols = [col for col in categorical_cols if col in data.columns]

    # Check if we have enough features
    if not numerical_cols:
        st.error("No numerical features available for clustering.")
        return

    # Create features DataFrame
    features = data[numerical_cols].copy()

    # Handle categorical columns with one-hot encoding
    if categorical_cols:
        features = pd.concat([features, pd.get_dummies(data[categorical_cols], drop_first=True)], axis=1)

    # Ensure no non-numerical columns (e.g., Date, Store ID, Product ID) are included
    features = features.select_dtypes(include=['float64', 'int64', 'uint8'])

    # Handle missing values
    if features.isnull().any().any():
        st.warning("Missing values detected. Filling with zeros.")
        features = features.fillna(0)

    # Scale features for clustering
    scaler = StandardScaler()
    features_scaled = scaler.fit_transform(features)

    # Clustering method selection
    cluster_method = st.selectbox("Select Clustering Method", ["GMM", "t-SNE Visualization"])

    if cluster_method == "GMM":
        # Apply Gaussian Mixture Model with 3 clusters
        gmm = GaussianMixture(n_components=3, random_state=42)
        data['GMM_Cluster'] = gmm.fit_predict(features_scaled)

        # Calculate silhouette and Davies-Bouldin scores
        silhouette_gmm = silhouette_score(features_scaled, data['GMM_Cluster'])
        dbi_gmm = davies_bouldin_score(features_scaled, data['GMM_Cluster'])
        st.write(f"Silhouette Score (GMM): {silhouette_gmm:.3f}")
        st.write(f"Davies-Bouldin Index (GMM): {dbi_gmm:.3f}")

        # PCA visualization for GMM clusters
        pca = PCA(n_components=2)
        pca_result = pca.fit_transform(features_scaled)
        data['PCA1'] = pca_result[:, 0]
        data['PCA2'] = pca_result[:, 1]

        fig, ax = plt.subplots(figsize=(7, 6))
        sns.scatterplot(data=data, x='PCA1', y='PCA2', hue='GMM_Cluster', palette='viridis', s=60, alpha=0.7, ax=ax)
        ax.set_title("PCA Visualization of GMM Clusters")
        ax.set_xlabel("PCA Component 1")
        ax.set_ylabel("PCA Component 2")
        ax.grid(True, linestyle='--', alpha=0.5)
        st.pyplot(fig)

    elif cluster_method == "t-SNE Visualization":
        if 'GMM_Cluster' not in data.columns:
            st.warning("Please run GMM clustering first to generate cluster labels for t-SNE visualization.")
            return

        # Apply t-SNE for 2D visualization
        tsne = TSNE(n_components=2, random_state=42)
        tsne_result = tsne.fit_transform(features_scaled)
        data['tSNE1'] = tsne_result[:, 0]
        data['tSNE2'] = tsne_result[:, 1]

        # Plot t-SNE visualization for GMM clusters
        fig, ax = plt.subplots(figsize=(7, 6))
        sns.scatterplot(data=data, x='tSNE1', y='tSNE2', hue='GMM_Cluster', palette='viridis', s=60, alpha=0.7, ax=ax)
        ax.set_title("t-SNE Visualization of GMM Clusters")
        ax.set_xlabel("t-SNE Component 1")
        ax.set_ylabel("t-SNE Component 2")
        ax.grid(True, linestyle='--', alpha=0.5)
        st.pyplot(fig)