import pickle
import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
from streamlit_option_menu import option_menu

# Load models
heart_disease_model = pickle.load(open('model/heart_disease_model.sav', 'rb'))
lung_cancer_model = pickle.load(open('model/lung_cancer_model.sav', 'rb'))

# Sidebar navigation
with st.sidebar:
    selected = option_menu(
        'Health App Menu',
        ['Heart Disease Predictor', 'Heart Data Visualizer', 'Lung Cancer Predictor', 'Lung Data Visualizer'],
        icons=['heart', 'bar-chart', 'lungs'],
        default_index=0
    )

if selected == 'Heart Disease Predictor':
    st.title('❤️ Heart Disease Prediction using ML')

    col1, col2, col3 = st.columns(3)

    with col1:
        age = st.slider('Age', min_value=29, max_value=77, value=50)
        sex = st.selectbox('Sex (0 or 1)', options=[0, 1])
        cp = st.selectbox('Chest Pain Type (0–3)', options=[0, 1, 2, 3])
        fbs = st.selectbox('Fasting Blood Sugar > 120 (0 or 1)', options=[0, 1])
        restecg = st.selectbox('Resting ECG (0–2)', options=[0, 1, 2])

    with col2:
        trestbps = st.slider('Resting BP (mm Hg)', min_value=94, max_value=200, value=120)
        chol = st.slider('Cholesterol (mg/dl)', min_value=100, max_value=600, value=200)
        thalach = st.slider('Max Heart Rate', min_value=70, max_value=202, value=150)
        exang = st.selectbox('Exercise Induced Angina (0 or 1)', options=[0, 1])
        slope = st.selectbox('Slope (0–2)', options=[0, 1, 2])

    with col3:
        oldpeak = st.slider('Oldpeak (ST depression)', min_value=0.0, max_value=6.2, value=1.0, step=0.1)
        ca = st.selectbox('Number of Vessels (0–4)', options=[0, 1, 2, 3, 4])
        thal = st.selectbox('Thalassemia (0–3)', options=[0, 1, 2, 3])

    heart_diagnosis = ''

    if st.button('Heart Disease Test Result'):
        try:
            input_data = [
                age, sex, cp, trestbps, chol, fbs, restecg,
                thalach, exang, oldpeak, slope, ca, thal
            ]

            prediction = heart_disease_model.predict([input_data])

            if prediction[0] == 1:
                heart_diagnosis = '⚠️ The person **has** heart disease.'
            else:
                heart_diagnosis = '✅ The person **does not** have heart disease.'

            st.success(heart_diagnosis)
        except Exception as e:
            st.error(f"Prediction failed: {e}")


# --- Heart Data Visualizer ---
elif selected == 'Heart Data Visualizer':
    st.title("📊 Explore & Visualize Heart Dataset")

    try:
        df = pd.read_csv('data_preprocessed/heart.csv')
        st.subheader("Dataset Preview")
        st.dataframe(df.head())

        vis_type = st.selectbox("Select Visualization Type", ['Univariate', 'Bivariate', 'Multivariate'])

        if vis_type == 'Univariate':
            col = st.selectbox("Select a column", df.columns)
            plot_type = st.radio("Plot Type", ['Histogram', 'Boxplot', 'Line Chart'])

            st.subheader(f"{plot_type} of {col}")
            fig, ax = plt.subplots()
            if plot_type == 'Histogram':
                ax.hist(df[col].dropna(), bins=20, color='skyblue', edgecolor='black')
            elif plot_type == 'Boxplot':
                ax.boxplot(df[col].dropna())
            elif plot_type == 'Line Chart':
                ax.plot(df[col].dropna(), color='green')
            ax.set_title(f"{plot_type} of {col}")
            st.pyplot(fig)

        elif vis_type == 'Bivariate':
            x_col = st.selectbox("Select X-axis", df.columns, key='biv_x')
            y_col = st.selectbox("Select Y-axis", df.columns, key='biv_y')
            plot_type = st.radio("Plot Type", ['Scatter Plot', 'Line Plot'])

            st.subheader(f"{plot_type} of {x_col} vs {y_col}")
            fig, ax = plt.subplots()
            if plot_type == 'Scatter Plot':
                ax.scatter(df[x_col], df[y_col], alpha=0.7, color='purple')
            elif plot_type == 'Line Plot':
                ax.plot(df[x_col], df[y_col], color='blue')
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
        st.error("⚠️ 'heart.csv' not found. Please ensure it's in the 'data_preprocessed/' folder.")

# --- Lung Cancer Prediction ---
elif selected == 'Lung Cancer Predictor':
    st.title('🫁 Lung Cancer Prediction using ML')

    col1, col2, col3 = st.columns(3)

    with col1:
        gender = st.selectbox('Gender (0=Female, 1=Male)', [0, 1])
        smoking = st.selectbox('Smoking (0=No, 1=Yes, 2=Heavily)', [0, 1, 2])
        yellow_fingers = st.selectbox('Yellow Fingers', [0, 1])
        chronic_disease = st.selectbox('Chronic Disease (0=None, 1=Mild, 2=Severe)', [0, 1, 2])
        allergy = st.selectbox('Allergy', [0, 1])

    with col2:
        age = st.slider('Age (scaled 0 to 1)', 0.0, 1.0, 0.5, 0.01)
        anxiety = st.selectbox('Anxiety', [0, 1])
        peer_pressure = st.selectbox('Peer Pressure', [0, 1])
        fatigue = st.selectbox('Fatigue', [0, 1])
        alcohol = st.selectbox('Alcohol Consuming', [0, 1])

    with col3:
        wheezing = st.selectbox('Wheezing', [0, 1])
        coughing = st.selectbox('Coughing', [0, 1])
        shortness_of_breath = st.selectbox('Shortness of Breath', [0, 1])
        swallowing_difficulty = st.selectbox('Swallowing Difficulty', [0, 1])
        chest_pain = st.selectbox('Chest Pain', [0, 1])

    lung_diagnosis = ''

    if st.button('Lung Cancer Test Result'):
        input_data = [
            gender, age, smoking, yellow_fingers, anxiety, peer_pressure,
            chronic_disease, fatigue, allergy, wheezing,
            alcohol, coughing, shortness_of_breath,
            swallowing_difficulty, chest_pain
        ]
        try:
            prediction = lung_cancer_model.predict([input_data])

            if prediction[0] == 1:
                lung_diagnosis = '⚠️ The person may have lung cancer.'
            else:
                lung_diagnosis = '✅ The person does not show signs of lung cancer.'

            st.success(lung_diagnosis)
        except Exception as e:
            st.error(f"Prediction failed: {e}")

# --- Lung Data Visualizer ---
elif selected == 'Lung Data Visualizer':
    st.title("📊 Explore & Visualize Lung Cancer Dataset")

    try:
        df = pd.read_csv('data_preprocessed/lung.csv')  # Ensure this file exists

        st.subheader("Dataset Preview")
        st.dataframe(df.head())

        vis_type = st.selectbox("Select Visualization Type", ['Univariate', 'Bivariate', 'Multivariate'])

        if vis_type == 'Univariate':
            col = st.selectbox("Select a column", df.columns)
            plot_type = st.radio("Plot Type", ['Histogram', 'Boxplot', 'Line Chart'])

            st.subheader(f"{plot_type} of {col}")
            fig, ax = plt.subplots()
            if plot_type == 'Histogram':
                ax.hist(df[col].dropna(), bins=20, color='orange', edgecolor='black')
            elif plot_type == 'Boxplot':
                ax.boxplot(df[col].dropna())
            elif plot_type == 'Line Chart':
                ax.plot(df[col].dropna(), color='brown')
            ax.set_title(f"{plot_type} of {col}")
            st.pyplot(fig)

        elif vis_type == 'Bivariate':
            x_col = st.selectbox("Select X-axis", df.columns, key='lung_biv_x')
            y_col = st.selectbox("Select Y-axis", df.columns, key='lung_biv_y')
            plot_type = st.radio("Plot Type", ['Scatter Plot', 'Line Plot'])

            st.subheader(f"{plot_type} of {x_col} vs {y_col}")
            fig, ax = plt.subplots()
            if plot_type == 'Scatter Plot':
                ax.scatter(df[x_col], df[y_col], alpha=0.7, color='teal')
            elif plot_type == 'Line Plot':
                ax.plot(df[x_col], df[y_col], color='gray')
            ax.set_xlabel(x_col)
            ax.set_ylabel(y_col)
            ax.set_title(f"{x_col} vs {y_col}")
            st.pyplot(fig)

        elif vis_type == 'Multivariate':
            st.subheader("Correlation Heatmap")
            corr = df.corr(numeric_only=True)
            fig, ax = plt.subplots()
            im = ax.imshow(corr, cmap='YlGnBu', interpolation='nearest')
            ax.set_xticks(range(len(corr.columns)))
            ax.set_yticks(range(len(corr.columns)))
            ax.set_xticklabels(corr.columns, rotation=90)
            ax.set_yticklabels(corr.columns)
            fig.colorbar(im, ax=ax)
            st.pyplot(fig)

    except FileNotFoundError:
        st.error("⚠️ 'lung.csv' not found. Please ensure it's in the 'data_preprocessed/' folder.")

