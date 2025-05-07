import pickle
import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
from streamlit_option_menu import option_menu

# Load model
heart_disease_model = pickle.load(open('model/heart_disease_model.sav', 'rb'))

# Sidebar navigation
with st.sidebar:
    selected = option_menu(
        'Health App Menu',
        ['Heart Disease Predictor', 'Heart Data Visualizer'],
        icons=['heart', 'bar-chart'],
        default_index=0
    )

# --- Heart Disease Prediction ---
if selected == 'Heart Disease Predictor':
    st.title('Heart Disease Prediction using ML')

    col1, col2, col3, col4 = st.columns(4)

    with col1:
        age = st.text_input('Age')
    with col2:
        sex = st.text_input('Sex: 0 or 1')
    with col3:
        cp = st.text_input('Chest Pain types: 0-3')
    with col4:
        trestbps = st.text_input('Resting BP')

    with col1:
        chol = st.text_input('Cholesterol (mg/dl)')
    with col2:
        fbs = st.text_input('Fasting Blood Sugar: 0 or 1')
    with col3:
        restecg = st.text_input('Resting ECG: 0-2')
    with col4:
        thalach = st.text_input('Max Heart Rate')

    with col1:
        exang = st.text_input('Exercise Induced Angina: 0 or 1')
    with col2:
        oldpeak = st.text_input('ST Depression')
    with col3:
        slope = st.text_input('Slope: 0-2')
    with col4:
        ca = st.text_input('Vessels Colored (0-4)')

    with col1:
        thal = st.text_input('Thal: 0-3')

    heart_diagnosis = ''

    if st.button('Heart Disease Test Result'):
        if not all([age, sex, cp, trestbps, chol, fbs, restecg,
                    thalach, exang, oldpeak, slope, ca, thal]):
            st.error('Please fill out all input fields.')
        else:
            try:
                input_data = [float(age), float(sex), float(cp), float(trestbps), float(chol),
                              float(fbs), float(restecg), float(thalach), float(exang),
                              float(oldpeak), float(slope), float(ca), float(thal)]

                prediction = heart_disease_model.predict([input_data])

                if prediction[0] == 1:
                    heart_diagnosis = 'The person has heart disease.'
                else:
                    heart_diagnosis = 'The person does not have heart disease.'

                st.success(heart_diagnosis)
            except ValueError:
                st.error("Please enter valid numeric inputs.")

# --- Health Data Visualizer ---
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
        st.error("⚠️ 'data.csv' not found. Please ensure it's in the project folder.")
