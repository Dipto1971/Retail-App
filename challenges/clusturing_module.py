import streamlit as st
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.mixture import GaussianMixture
from sklearn.manifold import TSNE
from sklearn.metrics import silhouette_score, davies_bouldin_score

def run_inventory_clustering(data):
    st.header("Challenge 3: Clustering for Inventory Insights")
    st.write("This section uses clustering to identify inventory patterns.")

    # Clustering method selection
    cluster_method = st.selectbox("Select Clustering Method", [
                                  "GMM", "t-SNE Visualization"])

    features = data[['Inventory Level', 'Units Sold', 'Demand Forecast']]

    if cluster_method == "GMM":
        gmm = GaussianMixture(n_components=3, random_state=42)
        data['GMM_Cluster'] = gmm.fit_predict(features)

        # Metrics
        silhouette = silhouette_score(features, data['GMM_Cluster'])
        dbi = davies_bouldin_score(features, data['GMM_Cluster'])
        st.write(f"Silhouette Score: {silhouette:.3f}")
        st.write(f"Davies-Bouldin Index: {dbi:.3f}")

        # PCA-like visualization
        fig, ax = plt.subplots()
        sns.scatterplot(data=data, x='Inventory Level',
                        y='Units Sold', hue='GMM_Cluster', palette='viridis', ax=ax)
        ax.set_title("GMM Clustering - Inventory vs Units Sold")
        ax.grid(True, linestyle='--', alpha=0.5)
        st.pyplot(fig)

    elif cluster_method == "t-SNE Visualization":
        if 'GMM_Cluster' not in data.columns:
            st.warning("Run GMM clustering first to see cluster labels.")
            return

        tsne = TSNE(n_components=2, random_state=42)
        tsne_result = tsne.fit_transform(features)
        data['tSNE1'] = tsne_result[:, 0]
        data['tSNE2'] = tsne_result[:, 1]

        fig, ax = plt.subplots()
        sns.scatterplot(data=data, x='tSNE1', y='tSNE2',
                        hue='GMM_Cluster', palette='viridis', ax=ax)
        ax.set_title("t-SNE Visualization of GMM Clusters")
        ax.grid(True, linestyle='--', alpha=0.5)
        st.pyplot(fig)
