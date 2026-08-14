import numpy as np
import streamlit as st
import pandas as pd
from sklearn import datasets
from sklearn.tree import DecisionTreeClassifier


st.write(''' # Predicción de depresion con uso de Redes Sociales y Hábitos  ''')
st.image("SOCIALMEDIA.jpg", caption="Conocer tus datos de cuanto tiempo pasas en redes sociales, algunos hábitos digitales y hábitos de la vida cotidiana te ayudarán a prevenir o disminuir los efectos de la depresión.")

st.header('Hablemos de tus hábitos...')

def user_input_features():
  # Entrada
  age = st.number_input('Edad en años:', min_value=0, max_value=3, value = 1, step = 1)
  daily_social_media_hours = st.number_input('Tiempo diario en redes sociales:', min_value=0, max_value=24, value = 0, step = 0.5)
  sleep_hours = st.number_input('Tiempo de sueño al día:', min_value=0, max_value=24, value = 5, step = 0.5)
  screen_time_before_sleep = st.number_input('Tiempo en pantalla antes de dormir en horas:',min_value=0, max_value=24, value = 2, step = 0.5)
  academic_performance = st.number_input('Desempeño escolar del 1 al 10:', min_value=0, max_value=10, value = 0, step = 0.5)
  physical_activity = st.number_input('Actividad física en horas:',min_value=0, max_value=10, value = 1, step = 0.5)
  stress_level = st.number_input('Nivel de estrés del 1 al 10:', min_value=0, max_value=10, value = 5, step = .5)
  anxiety_level = st.number_input('Nivel de ansiedad del 1 al 10:', min_value=0, max_value=10, value = 5, step = .5)
  addiction_level = st.number_input('Nivel de adición del 1 al 10:', min_value=0, max_value=10, value = 5, step = .5)
  gender_male = st.number_input('Sexo: male=1 female=0', min_value=0, max_value=1, value = 0, step = 1)
  platform_usage_Instagram = st.number_input('Uso de instagram: 1=si 0=no', min_value=0, max_value=1, value = 0, step = 1)
  platform_usage_TikTok = st.number_input('Uso de TikTok: 1=si 0=no', min_value=0, max_value=1, value = 0, step = 1)
  social_interaction_level_low = st.number_input('Nivel de interacción social bajo: 1=si 0=no', min_value=0, max_value=1, value = 0, step = 1)
  social_interaction_level_medium = st.number_input('Nivel de interaccion social medio: 1=si 0=no'min_value=0, max_value=1, value = 0, step = 1)


  user_input_data = {'age':age,
                     'daily_social_media_hours':daily_social_media_hours,
                     'sleep_hours':sleep_hours,
                     'screen_time_before_sleep':screen_time_before_sleep,
                     'academic_performance':academic_performance, 
                     'physical_activity':physical_activity,
                     'stress_level':stress_level,
                     'anxiety_level':anxiety_level,
                     'addiction_level':addiction_level,
                     'gender_male':gender_male,      
                     'platform_usage_Instagram':platform_usage_Instagram,
                     'platform_usage_TikTok':platform_usage_TikTok,
                     'social_interaction_level_low':social_interaction_level_low, 
                     'social_interaction_level_medium':social_interaction_level_medium}


  features = pd.DataFrame(user_input_data, index=[0])

  return features

df = user_input_features()

socmedhea =  pd.read_csv('Teen_Mental_Health_Dataset.csv', encoding='latin-1')
X = socmedhea.drop(columns='depression_label')
Y = socmedhea['depression_label']
#este es en codigo de ejemplo, debajo aplico el mio para probar si funciona
#classifier = DecisionTreeClassifier(max_depth=8, criterion='entropy', min_samples_leaf=5, max_features=7, random_state=0)
classifier=DecisionTreeClassifier(max_depth=4,min_samples_split=5,random_state=42)
classifier.fit(X, Y)

prediction = classifier.predict(df)

st.subheader('Predicción')
if prediction == 0:
  st.write('No hay riesgo por depresión')
elif prediction == 1:
  st.write('Existe presencia de depresión')
else:
  st.write('Sin predicción')
