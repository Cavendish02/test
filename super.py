import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
import shap
import matplotlib.pyplot as plt
from pycaret.regression import setup, compare_models, finalize_model, predict_model

# تعريف البيانات
data_url = "http://lib.stat.cmu.edu/datasets/boston"
raw_df = pd.read_csv(data_url, sep="\s+", skiprows=22, header=None)

# تسمية الأعمدة
feature_names = [
    "CRIM", "ZN", "INDUS", "CHAS", "NOX", "RM", "AGE", "DIS", "RAD", "TAX",
    "PTRATIO", "B", "LSTAT", "MEDV"
]
raw_df.columns = feature_names

# تقسيم البيانات
X = raw_df.drop(columns=["MEDV"])
y = raw_df["MEDV"]
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# إعداد النموذج باستخدام pycaret
regression_setup = setup(data=raw_df, target="MEDV", silent=True)
best_model = compare_models()
final_model = finalize_model(best_model)

# التنبؤ باستخدام النموذج النهائي
predictions = predict_model(final_model, data=X_test)

# تقييم النموذج
from sklearn.metrics import mean_squared_error
mse = mean_squared_error(y_test, predictions["Label"])
print("Mean Squared Error:", mse)

# شرح أهمية السمات
explainer = shap.TreeExplainer(final_model)
shap_values = explainer.shap_values(X)
shap.summary_plot(shap_values, X, feature_names=X.columns)

# العرض التفاعلي باستخدام Streamlit
import streamlit as st
st.header("تقييم النموذج")
st.write("Mean Squared Error:", mse)
st.write("---")
st.header("شرح أهمية السمات")
st.pyplot(shap.summary_plot(shap_values, X, feature_names=X.columns))
