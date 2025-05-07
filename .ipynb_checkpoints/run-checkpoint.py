import pickle
import streamlit as st
from streamlit_option_menu import option_menu


# loading the saved models
heart_disease_model = pickle.load(open('model/heart_disease_model.sav','rb'))

with st.sidebar:
    
    selected = option_menu('Disease Predictor',
                          
                          [
                           'Heart Disease'],
                          icons=['heart'],
                          default_index=0)
    # Heart Disease Prediction Page
if (selected == 'Heart Disease'):
    
    # page title
    st.title('Heart Disease Prediction using ML')
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        age = st.text_input('Enter Age')
        
    with col2:
        sex = st.text_input('Sex: 0 and 1')
        
    with col3:
        cp = st.text_input('Chest Pain types: 0, 1, 2, 3')
        
    with col1:
        trestbps = st.text_input('Resting Blood Pressure eg: 145')
        
    with col2:
        chol = st.text_input('Serum Cholestoral in mg/dl eg: 233')
        
    with col3:
        fbs = st.text_input('Fasting Blood Sugar > 120 mg/dl: 0 and 1')
        
    with col1:
        restecg = st.text_input('Resting Electrocardiographic: 0, 1, 2')
        
    with col2:
        thalach = st.text_input('Maximum Heart Rate achieved eg: 150')
        
    with col3:
        exang = st.text_input('Exercise Induced Angina : 0, 1')
        
    with col1:
        oldpeak = st.text_input('ST depression induced by exercise eg: 2.3, 1.5')
        
    with col2:
        slope = st.text_input('Slope of the peak exercise ST segment: 0, 1, 2')
        
    with col3:
        ca = st.text_input('Major vessels colored by flourosopy 0, 1, 2, 3, 4')
        
    with col1:
        thal = st.text_input('eg. 1 , 2     thal: 0 = normal; 1 = fixed defect; 2 = reversable defect')
        
        
     
     
    # code for Prediction
    heart_diagnosis = ''
    
    
    # creating a button for Prediction
    
    if st.button('Heart Disease Test Result'):
        # Convert input values to numeric format
        age = float(age)
        sex = float(sex)
        cp = float(cp)
        trestbps = float(trestbps)
        chol = float(chol)
        fbs = float(fbs)
        restecg = float(restecg)
        thalach = float(thalach)
        exang = float(exang)
        oldpeak = float(oldpeak)
        slope = float(slope)
        ca = float(ca)
        thal = float(thal)
        heart_prediction = heart_disease_model.predict([[age, sex, cp, trestbps, chol, fbs, restecg,thalach,exang,oldpeak,slope,ca,thal]])                          
        
        if (heart_prediction[0] == 1):
          heart_diagnosis = 'The person is having heart disease'
        else:
          heart_diagnosis = 'The person does not have any heart disease'
        
    st.success(heart_diagnosis)
        
    
