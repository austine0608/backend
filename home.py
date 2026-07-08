import streamlit as st 
import numpy as np 
import pandas as pd 
from sklearn.tree import DecisionTreeClassifier 
from sklearn.model_selection import train_test_split 
from sklearn.metrics import accuracy_score 

df = pd.read_csv('music.csv') 

with st.expander('Decision Tree Machine Model'):
    st.dataframe(df) 

st.image('./assets/1.png')

x = df[['age','gender']] 
y = df[['genre']] 

x_train,x_test,y_train,y_test = train_test_split(x,y,test_size=0.2)  

model = DecisionTreeClassifier() 
model.fit(x,y)

predictions = model.predict(x_test) 

score = accuracy_score(y_test,predictions) 
score 


age = st.slider('Select age range',10,70,100) 
gender = st.selectbox('Select Gender', options=['Male','Female']) 

gender_code = 1 if gender == 'Male' else 0 

if st.button('Click To Get Classification'):
    prediction = model.predict([[age,gender_code]])
    st.success(f'Correct: **{prediction[0]}**') 

st.video('./assets/1.mp4')