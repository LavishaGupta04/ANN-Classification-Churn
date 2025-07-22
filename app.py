import pandas as pd
import numpy as np
import streamlit as st
import tensorflow as tf
from sklearn.preprocessing import StandardScaler,OneHotEncoder,LabelEncoder
import pickle

#Load the model
model=tf.keras.models.load_model('model.h5')

#Load the pickle files
with open('label_en_gender.pkl','rb') as file:
    label_en_gender=pickle.load(file)

with open('scaler.pkl','rb') as file:
    scaler=pickle.load(file)

with open('OneHot_encoder_geo.pkl','rb') as file:
    OneHot_encoder_geo=pickle.load(file)

#Sreamlit app
st.title('Customer Churn Prediction')

#User Input
geography=st.selectbox('Geography',OneHot_encoder_geo.categories_[0])
age=st.slider("Age",18,92)
gender=st.selectbox("Gender",label_en_gender.classes_)
balance=st.number_input('Balance')
credit_score=st.number_input('Credit Score')
estimated_salary=st.number_input('Estimated Salary')
tenure=st.slider("Tenure",0,10)
num_of_products=st.slider('Number Of Products',0,4)
has_cr_card=st.selectbox('Has Credit Card',[0,1])
is_active_member=st.selectbox('Is Active member',[0,1])

#Prepare the input data
input_data=pd.DataFrame({
    'CreditScore':[credit_score],
    #'Geography':'Spain',
    'Gender':[label_en_gender.transform([gender])[0]],
    'Age':[age],
    'Tenure':[tenure],
    'Balance':[balance],
    'NumOfProducts':[num_of_products],
    'HasCrCard':[has_cr_card],
    'IsActiveMember':[is_active_member],
    'EstimatedSalary':[estimated_salary]

})

#One Hot Encode 'Geography'
geo_encoded=OneHot_encoder_geo.transform([[geography]]).toarray()
geo_encoded_df=pd.DataFrame(geo_encoded,columns=OneHot_encoder_geo.get_feature_names_out(['Geography']))

#Combining the data
input_data=pd.concat([input_data.reset_index(drop=True),geo_encoded_df],axis=1)

#Scale tye data
sacled_data=scaler.transform(input_data)

#Predict the churn
prediction=model.predict(sacled_data)
prediction_prob=prediction[0][0]
st.write(f"The churn probability is : {prediction_prob:.2f}")

if prediction_prob>0.5:
    st.write("The Customer is likely to churn")
else:
    st.write("He will not churn")





