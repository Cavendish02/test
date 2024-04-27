#!/usr/bin/env python
# coding: utf-8

# In[2]:


#!/usr/bin/env python
# coding: utf-8

get_ipython().system('pip install streamlit')
get_ipython().system('pip install pandas numpy scikit-learn shap')

import streamlit as st
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
import shap
import matplotlib.pyplot as plt
from pycaret.regression import setup, compare_models, finalize_model, predict_model

data_url = "http://lib.stat.cmu.edu/datasets/boston"
raw_df = pd.read_csv(data_url, sep="\s+", skiprows=22, header=None)
data = np.hstack([raw_df.values[::2, :], raw_df.values[1::2, :2]])
target = raw_df.values[1::2, 2]

X = pd.DataFrame(data, columns=["CRIM", "ZN", "INDUS", "CHAS", "NOX", "RM", "AGE", "DIS", "RAD", "TAX", "PTRATIO", "B", "LSTAT"])
Y = pd.DataFrame(target, columns=["MEDV"])

st.sidebar.header('Specify Input Parameters')

def user_input_features():
    features = {}
    for feature in X.columns:
        features[feature] = st.sidebar.slider(feature, X[feature].min(), X[feature].max(), X[feature].mean())
    return pd.DataFrame(features, index=[0])

df = user_input_features()

st.header('Specified Input parameters')
st.write(df)
st.write('---')

regression_setup = setup(data=pd.concat([X, Y], axis=1), target='MEDV')
best_model = compare_models()
final_model = finalize_model(best_model)
final_model = final_model[-1]  # Extract the last component of the pipeline (which is the model)
prediction = predict_model(final_model, data=df)

st.header('Prediction of MEDV')
st.write(prediction)
st.write('---')

explainer = shap.TreeExplainer(final_model)
shap_values = explainer.shap_values(X)

st.header('Feature Importance')
plt.title('Feature importance based on SHAP values')
shap.summary_plot(shap_values, X)
st.pyplot(bbox_inches='tight')
st.write('---')

plt.title('Feature importance based on SHAP values (Bar)')
shap.summary_plot(shap_values, X, plot_type="bar")
st.pyplot(bbox_inches='tight')


# In[ ]:




