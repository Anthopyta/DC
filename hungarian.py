import itertools
import pandas as pd
import numpy as np
from imblearn.over_sampling import SMOTE
from sklearn.metrics import accuracy_score
import streamlit as st
import time
import pickle

with open("data/hungarian.data", encoding='Latin1') as file:
  lines = [line.strip() for line in file]
  
data = itertools.takewhile(
  lambda x: len(x) == 76,
  (' '.join(lines[i: (i + 10)]).split() for i in range(0, len(lines),10))
)

df = pd.DataFrame.from_records(data)

df = df.iloc[:, :-1]
df = df.drop(df.columns[0], axis=1)
df = df.astype(float)

df.replace(-9.0, np.NaN, inplace=True)

df_selected = df.iloc[:, [1, 2, 7, 8, 10, 14, 17, 30, 36, 38, 39, 42, 49, 56]]

column_mapping = {
  2: 'age', 
  3: 'sex',
  8: 'cp', 
  9: 'trestbps',
  11: 'chol',
  15: 'fbs',
  18: 'restecg',
  31: 'thalach',
  37: 'exang',
  39: 'oldpeak',
  40: 'slope',
  43: 'ca',
  50: 'thal',
  57: 'target'
}

df_selected.rename(columns=column_mapping, inplace=True)

columns_to_drop = ['ca', 'slope', 'thal']
df_selected = df_selected.drop(columns_to_drop, axis=1)

meanTBPS = df_selected['trestbps'].dropna()
meanChol = df_selected['chol'].dropna()
meanfbs = df_selected['fbs'].dropna()
meanRestCG = df_selected['restecg'].dropna()
meanthalach = df_selected['thalach'].dropna()
meanexang = df_selected['exang'].dropna()

meanTBPS = meanTBPS.astype(float)
meanChol = meanChol.astype(float)
meanfbs = meanfbs.astype(float)
meanthalach = meanthalach.astype(float)
meanexang = meanexang.astype(float)
meanRestCG = meanRestCG.astype(float)

meanTBPS = round(meanTBPS.mean())  
meanChol = round(meanChol.mean())  
meanfbs = round(meanfbs.mean()) 
meanthalach = round(meanthalach.mean()) 
meanexang = round(meanexang.mean()) 
meanRestCG = round(meanRestCG.mean())  

fill_values = {
  'trestbps': meanTBPS,
  'chol': meanChol,
  'fbs': meanfbs,
  'thalach': meanthalach, 
  'exang': meanexang, 
  'restecg': meanRestCG
}
df_clean = df_selected.fillna(value=fill_values)
df_clean.drop_duplicates(inplace=True)

X = df_clean.drop('target', axis=1)
y = df_clean['target']

smote = SMOTE(random_state = 42)
X, y = smote.fit_resample(X, y)

model = pickle.load(open('model/xgb_model.pkl', 'rb'))

y_pred = model.predict(X)
accuracy = accuracy_score(y, y_pred)
accuracy = round((accuracy * 100), 2)

df_final = X
df_final['target'] = y

# =============================================================================
# ==============================STREAMLIT======================================

# Function to make predictions
def make_prediction(model, inputs):
    prediction = model.predict(inputs)[0]
    return prediction

# Function to display prediction result
def display_result(prediction):
    if prediction == 0:
        return ":green[**Healthy**]"
    else:
        return f":orange[**Heart disease level {prediction}**]"

# Streamlit UI
st.set_page_config(
    page_title="Hungarian Heart Disease",
    page_icon=":heart:"
)

st.title("Hungarian Heart Disease Prediction")
st.write(f"**_Model's Accuracy_** :  :green[**{accuracy}**]% (:red[_Do not copy outright_])")
st.write("")
st.sidebar.header("**User Input** Sidebar")

# User Input for Single Prediction
with st.sidebar.expander("Single Prediction", expanded=True):
    # Age
    age = st.number_input(label=":white[**Age**]", min_value=df_final['age'].min(), max_value=df_final['age'].max())
    
    # Sex
    sex_sb = st.selectbox(label=":white[**Sex**]", options=["Male", "Female"])
    if sex_sb == "Male":
        sex = 1
    elif sex_sb == "Female":
        sex = 0

    # Chest Pain Type
    cp_sb = st.selectbox(label=":white[**Chest pain type**]", options=["Typical angina", "Atypical angina", "Non-anginal pain", "Asymptomatic"])
    if cp_sb == "Typical angina":
        cp = 1
    elif cp_sb == "Atypical angina":
        cp = 2
    elif cp_sb == "Non-anginal pain":
        cp = 3
    elif cp_sb == "Asymptomatic":
        cp = 4

    trestbps = st.number_input(label=":white[**Resting blood pressure** (in mm Hg on admission to the hospital)]", min_value=df_final['trestbps'].min(), max_value=df_final['trestbps'].max())
    chol = st.number_input(label=":white[**Serum cholestoral** (in mg/dl)]", min_value=df_final['chol'].min(), max_value=df_final['chol'].max())
    
    # Fasting Blood Sugar
    fbs_sb = st.selectbox(label=":white[**Fasting blood sugar > 120 mg/dl?**]", options=["False", "True"])
    if fbs_sb == "False":
        fbs = 0
    elif fbs_sb == "True":
        fbs = 1

    # Resting ECG
    restecg_sb = st.selectbox(label=":white[**Resting electrocardiographic results**]", options=["Normal", "Having ST-T wave abnormality", "Showing left ventricular hypertrophy"])
    if restecg_sb == "Normal":
        restecg = 0
    elif restecg_sb == "Having ST-T wave abnormality":
        restecg = 1
    elif restecg_sb == "Showing left ventricular hypertrophy":
        restecg = 2

    thalach = st.number_input(label=":white[**Maximum heart rate achieved**]", min_value=df_final['thalach'].min(), max_value=df_final['thalach'].max())
    
    # Exercise Induced Angina
    exang_sb = st.selectbox(label=":white[**Exercise induced angina?**]", options=["No", "Yes"])
    if exang_sb == "No":
        exang = 0
    elif exang_sb == "Yes":
        exang = 1

    oldpeak = st.number_input(label=":white[**ST depression induced by exercise relative to rest**]", min_value=df_final['oldpeak'].min(), max_value=df_final['oldpeak'].max())
    
data = {
    'Age': age,
    'Sex': sex_sb,
    'Chest pain type': cp_sb,
    'RPB': f"{trestbps} mm Hg",
    'Serum Cholestoral': f"{chol} mg/dl",
    'FBS > 120 mg/dl?': fbs_sb,
    'Resting ECG': restecg_sb,
    'Maximum heart rate': thalach,
    'Exercise induced angina?': exang_sb,
    'ST depression': oldpeak,
}

preview_df = pd.DataFrame(data, index=['input'])
st.header("User Input as DataFrame")
st.write("")
st.dataframe(preview_df.iloc[:, :6])
st.write("")
st.dataframe(preview_df.iloc[:, 6:])
st.write("")
    
result = ":white[-]"
predict_btn = st.button("**Predict**", type="primary")

if predict_btn:
    inputs = [[age, sex, cp, trestbps, chol, fbs, restecg, thalach, exang, oldpeak]]
    prediction = make_prediction(model, inputs)
    result = display_result(prediction)
    
    with st.spinner('Predicting...'):
        time.sleep(2)
    if prediction == 0:
        result = ":green[**Healthy**]"
    elif prediction == 1:
        result = ":yellow[**Heart disease level 1**]"
    elif prediction == 2:
        result = ":orange[**Heart disease level 2**]"
    elif prediction == 3:
        result = ":red[**Heart disease level 3**]"
    elif prediction == 4:
        result = ":red[**Heart disease level 4**]"
    
st.write("")
st.write("")
st.subheader("Prediction:")
st.subheader(result)

# User Input for Multi Prediction
with st.sidebar.expander("Multi Prediction", expanded=False):
    st.header("Multi Prediction")

    # Download CSV Example
    sample_csv = df_final.iloc[:5, :-1].to_csv(index=False).encode('utf-8')
    st.write("")
    st.download_button("Download CSV Example", data=sample_csv, file_name='sample_heart_disease_parameters.csv', mime='text/csv')

    # Upload CSV
    st.write("")
    st.write("")
    file_uploaded = st.file_uploader("Upload a CSV file", type='csv')

    if file_uploaded:
        uploaded_df = pd.read_csv(file_uploaded)
        prediction_arr = model.predict(uploaded_df)

        # Progress Bar
        bar = st.progress(0)
        status_text = st.empty()

        for i in range(1, 101):
            status_text.text(f"{i}% complete")
            bar.progress(i)
            time.sleep(0.01)

        result_arr = []

        for prediction in prediction_arr:
            result = "Healthy" if prediction == 0 else f"Heart disease level {prediction}"
            result_arr.append(result)

        uploaded_result = pd.DataFrame({'Prediction Result': result_arr})

        # Complete Progress Bar
        for i in range(70, 101):
            status_text.text(f"{i}% complete")
            bar.progress(i)
            time.sleep(0.01)
            if i == 100:
                time.sleep(1)
                status_text.empty()
                bar.empty()

        # Display results and uploaded data
        col1, col2 = st.columns([1, 2])

        with col1:
            st.dataframe(uploaded_result)
        with col2:
            st.dataframe(uploaded_df)


# Footer
st.sidebar.text("")
st.sidebar.markdown("**_Explore your heart health!_** :heart: :chart_with_upwards_trend:")


