import pickle
import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
from streamlit_option_menu import option_menu
import numpy as np

# Load models
heart_disease_model = pickle.load(open('model/heart_disease_model.sav', 'rb'))
lung_cancer_model = pickle.load(open('model/lung_cancer_model.sav', 'rb'))
diabetes_model = pickle.load(open('model/diabetes_disease_model.sav', 'rb'))
kidney_model = pickle.load(open('model/kidney_disease_model.sav', 'rb'))

# Sidebar navigation
with st.sidebar:
    selected = option_menu(
    'Health App Menu',
    [
        'Heart Disease Predictor',
        'Lung Cancer Predictor',
        'Diabetes Predictor',
        'Kidney Disease Predictor',
        'Dashboard'
    ],
    icons=['bar-chart','heart', 'lungs', 'activity', 'droplet', 'bar-chart'],
    default_index=0
)

if selected == 'Heart Disease Predictor':
    st.title('❤️ Heart Disease Prediction')

    col1, col2, col3 = st.columns(3)

    with col1:
        age = st.slider('Age', 29, 77, 50)
        sex = st.selectbox('Sex (0 = Female, 1 = Male)', [0, 1])
        cp = st.selectbox('Chest Pain Type (0-3)', [0, 1, 2, 3])
        fbs = st.selectbox('Fasting Blood Sugar > 120 mg/dl (0 = No, 1 = Yes)', [0, 1])
        restecg = st.selectbox('Resting ECG (0-2)', [0, 1, 2])

    with col2:
        trestbps = st.slider('Resting Blood Pressure (mm Hg)', 94, 200, 120)
        chol = st.slider('Cholesterol (mg/dl)', 100, 600, 200)
        thalach = st.slider('Max Heart Rate Achieved', 70, 202, 150)
        exang = st.selectbox('Exercise Induced Angina (0 = No, 1 = Yes)', [0, 1])
        slope = st.selectbox('Slope of ST Segment (0-2)', [0, 1, 2])

    with col3:
        oldpeak = st.slider('Oldpeak (ST depression)', 0.0, 6.2, 1.0, 0.1)
        ca = st.selectbox('Number of Major Vessels Colored by Fluoroscopy (0-4)', [0, 1, 2, 3, 4])
        thal = st.selectbox('Thalassemia (0 = Normal, 1 = Fixed Defect, 2 = Reversable Defect, 3 = Unknown)', [0, 1, 2, 3])

    if st.button('Heart Disease Test Result'):
        input_data = np.array([age, sex, cp, trestbps, chol, fbs, restecg,
                               thalach, exang, oldpeak, slope, ca, thal], dtype=float)

        try:
            prediction = heart_disease_model.predict([input_data])

            if prediction[0] == 1:
                st.error('⚠️ The person **has** heart disease.')
            else:
                st.success('✅ The person **does not** have heart disease.')
        except Exception as e:
            st.error(f"Prediction failed: {e}")


# --- Lung Cancer Prediction ---
import streamlit as st
import numpy as np
import pickle

# Load your trained model (adjust path)
# with open('lung_cancer_model.sav', 'rb') as f:
#     lung_cancer_model = pickle.load(f)

if selected == 'Lung Cancer Predictor':
    st.title('🫁 Lung Cancer Prediction')

    col1, col2, col3 = st.columns(3)

    with col1:
        gender = st.selectbox('Gender (0 = Female, 1 = Male)', [0, 1])
        smoking = st.selectbox('Smoking (0 = No, 1 = Yes, 2 = Heavily)', [0, 1, 2])
        yellow_fingers = st.selectbox('Yellow Fingers (0 = No, 1 = Yes)', [0, 1])
        chronic_disease = st.selectbox('Chronic Disease (0 = None, 1 = Mild, 2 = Severe)', [0, 1, 2])
        allergy = st.selectbox('Allergy (0 = No, 1 = Yes)', [0, 1])

    with col2:
        age = st.slider('Age (scaled 0.0 to 1.0)', 0.0, 1.0, 0.5, 0.01)
        anxiety = st.selectbox('Anxiety (0 = No, 1 = Yes)', [0, 1])
        peer_pressure = st.selectbox('Peer Pressure (0 = No, 1 = Yes)', [0, 1])
        fatigue = st.selectbox('Fatigue (0 = No, 1 = Yes)', [0, 1])
        alcohol = st.selectbox('Alcohol Consuming (0 = No, 1 = Yes)', [0, 1])

    with col3:
        wheezing = st.selectbox('Wheezing (0 = No, 1 = Yes)', [0, 1])
        coughing = st.selectbox('Coughing (0 = No, 1 = Yes)', [0, 1])
        shortness_of_breath = st.selectbox('Shortness of Breath (0 = No, 1 = Yes)', [0, 1])
        swallowing_difficulty = st.selectbox('Swallowing Difficulty (0 = No, 1 = Yes)', [0, 1])
        chest_pain = st.selectbox('Chest Pain (0 = No, 1 = Yes)', [0, 1])

    if st.button('Lung Cancer Test Result'):
        input_data = np.array([
            gender, age, smoking, yellow_fingers, anxiety, peer_pressure,
            chronic_disease, fatigue, allergy, wheezing,
            alcohol, coughing, shortness_of_breath,
            swallowing_difficulty, chest_pain
        ], dtype=float)


        try:
            prediction = lung_cancer_model.predict([input_data])

            if prediction[0] == 1:
                st.error('⚠️ The person may have lung cancer.')
            else:
                st.success('✅ The person does not show signs of lung cancer.')
        except Exception as e:
            st.error(f"Prediction failed: {e}")


elif selected == 'Diabetes Predictor':
    st.title('🩸 Diabetes Prediction')

    col1, col2 = st.columns(2)

    with col1:
        age = st.slider("Age (scaled)", -1.9, 1.8, 0.0, 0.01)
        hypertension = st.selectbox("Hypertension", [0, 1])
        heart_disease = st.selectbox("Heart Disease", [0, 1])
        bmi = st.slider("BMI (scaled)", -1.0, 5.0, 0.0, 0.01)

    with col2:
        hba1c = st.slider("HbA1c Level (scaled)", -2.0, 3.5, 0.0, 0.01)
        glucose = st.slider("Blood Glucose Level (scaled)", -1.5, 4.5, 0.0, 0.01)

        gender_male = st.checkbox("Male")
        gender_other = st.checkbox("Other")

    st.markdown("### Smoking History (One-Hot Encoded)")

    col3, col4, col5 = st.columns(3)
    with col3:
        smoking_current = st.checkbox("Current")
        smoking_ever = st.checkbox("Ever")
    with col4:
        smoking_former = st.checkbox("Former")
        smoking_never = st.checkbox("Never")
    with col5:
        smoking_not_current = st.checkbox("Not Current")

    if st.button('Diabetes Test Result'):
        input_data = np.array([
            age, hypertension, heart_disease, bmi, hba1c, glucose,
            int(gender_male), int(gender_other),
            int(smoking_current), int(smoking_ever), int(smoking_former),
            int(smoking_never), int(smoking_not_current)
        ], dtype=float)

        try:
            prediction = diabetes_model.predict([input_data])

            if prediction[0] == 1:
                st.error("⚠️ The person is likely to have diabetes.")
            else:
                st.success("✅ The person is unlikely to have diabetes.")
        except Exception as e:
            st.error(f"Prediction failed: {e}")


elif selected == 'Kidney Disease Predictor':
    st.title("🧠 Kidney Disease Prediction")

    col1, col2, col3 = st.columns(3)

    with col1:
        age = st.slider("Age", 20, 90, 50)
        gender = st.selectbox("Gender (0=Female, 1=Male)", [0, 1])
        prev_aki = st.selectbox("Previous Acute Kidney Injury", [0, 1])
        diastolic_bp = st.slider("Diastolic BP", 60, 120, 80)
        systolic_bp = st.slider("Systolic BP", 90, 180, 120)

    with col2:
        fasting_bs = st.number_input("Fasting Blood Sugar", min_value=0.0, max_value=300.0, value=100.0)
        serum_creatinine = st.number_input("Serum Creatinine", min_value=0.0, max_value=10.0, value=1.2)
        bun = st.number_input("BUN Levels", min_value=0.0, max_value=100.0, value=25.0)
        acr = st.number_input("Albumin-to-Creatinine Ratio (ACR)", min_value=0.0, max_value=500.0, value=100.0)

    with col3:
        gfr = st.number_input("GFR", min_value=0.0, max_value=120.0, value=60.0)
        edema = st.selectbox("Edema", [0, 1])
        protein_urine = st.number_input("Protein in Urine", min_value=0.0, max_value=5.0, value=1.0)
        nsaids_use = st.number_input("NSAIDs Use", min_value=0.0, max_value=10.0, value=5.0)

    if st.button("Kidney Disease Test Result"):
        try:
            input_data = [
                age, gender, prev_aki, diastolic_bp, fasting_bs,
                serum_creatinine, nsaids_use, edema, gfr,
                protein_urine, bun, acr, systolic_bp
            ]
            prediction = kidney_model.predict([input_data])
            if prediction[0] == 1:
                st.success("⚠️ The person **has** kidney disease.")
            else:
                st.success("✅ The person **does not** have kidney disease.")
        except Exception as e:
            st.error(f"Prediction failed: {e}")

elif selected == 'Dashboard':
    st.title("📊 Unified Health Data Visualizer")

    dataset_options = {
        'Heart': ('data_preprocessed/heart.csv', 'red'),
        'Lung': ('data_preprocessed/lung.csv', 'orange'),
        'Diabetes': ('data_preprocessed/diabetes.csv', 'green'),
        'Kidney': ('data_preprocessed/kidney.csv', 'blue'),
    }

    dataset_choice = st.selectbox("Select Dataset", list(dataset_options.keys()))
    file_path, color = dataset_options[dataset_choice]

    try:
        df = pd.read_csv(file_path)
        st.subheader(f"📁 Preview of {dataset_choice} Dataset")
        st.dataframe(df.head())

        vis_type = st.selectbox("Select Visualization Type", ['Univariate', 'Bivariate', 'Multivariate'])

        if vis_type == 'Univariate':
            col = st.selectbox("Select a column", df.columns)
            plot_type = st.radio("Plot Type", ['Histogram', 'Boxplot', 'Line Chart'])

            fig, ax = plt.subplots()
            if plot_type == 'Histogram':
                ax.hist(df[col].dropna(), bins=20, color=color, edgecolor='black')
            elif plot_type == 'Boxplot':
                ax.boxplot(df[col].dropna())
            elif plot_type == 'Line Chart':
                ax.plot(df[col].dropna(), color=color)
            ax.set_title(f"{plot_type} of {col}")
            st.pyplot(fig)

        elif vis_type == 'Bivariate':
            x_col = st.selectbox("Select X-axis", df.columns)
            y_col = st.selectbox("Select Y-axis", df.columns)
            plot_type = st.radio("Plot Type", ['Scatter Plot', 'Line Plot'])

            fig, ax = plt.subplots()
            if plot_type == 'Scatter Plot':
                ax.scatter(df[x_col], df[y_col], alpha=0.7, color=color)
            elif plot_type == 'Line Plot':
                ax.plot(df[x_col], df[y_col], color=color)
            ax.set_xlabel(x_col)
            ax.set_ylabel(y_col)
            ax.set_title(f"{x_col} vs {y_col}")
            st.pyplot(fig)

        elif vis_type == 'Multivariate':
            st.subheader("Correlation Heatmap")
            corr = df.corr(numeric_only=True)
            fig, ax = plt.subplots()
            im = ax.imshow(corr, cmap='coolwarm', interpolation='nearest')
            ax.set_xticks(range(len(corr.columns)))
            ax.set_yticks(range(len(corr.columns)))
            ax.set_xticklabels(corr.columns, rotation=90)
            ax.set_yticklabels(corr.columns)
            fig.colorbar(im, ax=ax)
            st.pyplot(fig)

    except FileNotFoundError:
        st.error(f"⚠️ '{file_path}' not found. Please ensure the dataset is in place.")
