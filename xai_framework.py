from abc import ABC, abstractmethod
import shap
from lime.lime_tabular import LimeTabularExplainer
from sklearn.inspection import partial_dependence, PartialDependenceDisplay
import matplotlib.pyplot as plt
import pandas as pd


class BaseExplainer(ABC):
    """
    Abstract base class for explainers.
    """
    def __init__(self, model, X_train: pd.DataFrame):
        self.model = model
        self.X_train = X_train
        self.feature_names = X_train.columns.tolist()

    @abstractmethod
    def explain(self, X: pd.DataFrame, **kwargs):
        """
        Generate explanation for X.
        """
        pass


class ShapExplainer(BaseExplainer):
    """
    SHAP-based explainer for tree and black-box models.
    """
    def __init__(self, model, X_train: pd.DataFrame):
        super().__init__(model, X_train)
        # Auto-detect tree vs kernel
        try:
            self.explainer = shap.TreeExplainer(self.model)
        except Exception:
            self.explainer = shap.KernelExplainer(self.model.predict, shap.sample(self.X_train, 100))

    def explain(self, X: pd.DataFrame, plot_type: str = "summary", **kwargs):
        shap_values = self.explainer.shap_values(X)
        fig = plt.figure(figsize=(10, 6))
        if plot_type == "summary":
            shap.summary_plot(shap_values, X, show=False)
        elif plot_type == "dependence":
            feature = kwargs.get("feature", self.feature_names[0])
            shap.dependence_plot(feature, shap_values, X, show=False)
        elif plot_type == "force":
            idx = kwargs.get("instance_idx", -1)
            shap.force_plot(self.explainer.expected_value, shap_values[idx], X.iloc[idx], matplotlib=True, show=False)
        return fig


class LimeExplainer(BaseExplainer):
    """
    LIME-based explainer for tabular data.
    """
    def __init__(self, model, X_train: pd.DataFrame):
        super().__init__(model, X_train)
        self.explainer = LimeTabularExplainer(
            training_data=self.X_train.values,
            feature_names=self.feature_names,
            mode="regression" if hasattr(model, "predict") else "classification"
        )

    def explain(self, X: pd.DataFrame, instance_idx: int = -1, num_features: int = 5):
        exp = self.explainer.explain_instance(
            X.iloc[instance_idx].values,
            self.model.predict,
            num_features=num_features
        )
        return exp


class PDPExplainer(BaseExplainer):
    """
    Partial Dependence Plot explainer.
    """
    def __init__(self, model, X_train: pd.DataFrame):
        super().__init__(model, X_train)

    def explain(self, X: pd.DataFrame, features: list = None):
        if features is None:
            features = self.feature_names[:2]
        fig, ax = plt.subplots(figsize=(10, 6))
        display = PartialDependenceDisplay.from_estimator(
            self.model, self.X_train, features, ax=ax
        )
        return fig


class ExplainerFactory:
    """
    Factory to create explainers by name.
    """
    @staticmethod
    def get_explainer(method: str, model, X_train: pd.DataFrame):
        method = method.lower()
        if method == "shap":
            return ShapExplainer(model, X_train)
        elif method == "lime":
            return LimeExplainer(model, X_train)
        elif method == "pdp":
            return PDPExplainer(model, X_train)
        else:
            raise ValueError(f"Unsupported explainer method: {method}")

# Example usage in your Streamlit app:
#
# import streamlit as st
# from xai_framework import ExplainerFactory
#
# X_train = df_features  # pd.DataFrame of training features
## model = models['forecast']
# method = st.selectbox('Choose XAI method', ['SHAP', 'LIME', 'PDP'])
# explainer = ExplainerFactory.get_explainer(method, model, X_train)
#
# if method == 'SHAP':
#     plot_type = st.selectbox('SHAP plot type', ['summary', 'dependence', 'force'])
#     params = {'plot_type': plot_type}
#     if plot_type == 'dependence':
#         params['feature'] = st.selectbox('Feature for dependence', X_train.columns)
#     if plot_type == 'force':
#         params['instance_idx'] = st.number_input('Instance index', 0, len(X_train)-1, len(X_train)-1)
#     fig = explainer.explain(X_train, **params)
#     st.pyplot(fig)
#
# elif method == 'LIME':
#     idx = st.number_input('Instance index', 0, len(X_train)-1, len(X_train)-1)
#     exp = explainer.explain(X_train, instance_idx=idx)
#     from streamlit.components.v1 import html
#     html(exp.as_html(), height=400)
#
# elif method == 'PDP':
#     features = st.multiselect('Features', X_train.columns, X_train.columns[:2])
#     fig = explainer.explain(X_train, features=features)
#     st.pyplot(fig)
