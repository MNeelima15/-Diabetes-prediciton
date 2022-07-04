from gettext import install
import pip
import streamlit as st
import pandas as pd 
from PIL import Image
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
from sklearn.model_selection import learning_curve, train_test_split
primaryColor="#1909a2"
backgroundColor="#fde5c6"
secondaryBackgroundColor="#ffcccc"
textColor="#0b0b0c"
font="serif"


df=pd.read_csv(r'C:\Users\ROOPA DATTA\Documents\Data science files\diabetes.csv')
df.describe()
st.title('Diabetes Detection')
image = Image.open('diabetes.jpg')

st.image(image)
with st.container():
 st.subheader("Training data")
 st.write(df.describe())
with st.container(): 
 st.subheader("Visualization")
 st.area_chart(df)
x=df.drop(["Outcome"],axis=1)
y=df.iloc[:,-1]
x.describe()
x_train, x_test, y_train, y_test = train_test_split(x,y, test_size = 0.2, random_state = 0)
def user_report():
  pregnancies = st.sidebar.number_input('Pregnancies', 0,17, 3 )
  glucose = st.sidebar.slider('Glucose', 0,200, 120 )
  bp = st.sidebar.slider('Blood Pressure', 0,122, 70 )
  skinthickness = st.sidebar.slider('Skin Thickness', 0,100, 20 )
  insulin = st.sidebar.slider('Insulin', 0,846, 79 )
  bmi = st.sidebar.slider('BMI', 0,67, 20 )
  dpf = st.sidebar.slider('Diabetes Pedigree Function', 0.0,2.4, 0.47 )
  age = st.sidebar.number_input('Age', 20,88, 33 )
 
  user_report = {
      'pregnancies':pregnancies,
      'glucose':glucose,
      'bp':bp,
      'skinthickness':skinthickness,
      'insulin':insulin,
      'bmi':bmi,
      'dpf':dpf,
      'age':age
  }
  report_data = pd.DataFrame(user_report, index=[0])
  return report_data

user_data = user_report()
lg= LogisticRegression()
lg.fit(x_train,y_train)
st.subheader('Accuracy: ')
st.write(str(accuracy_score(y_test, lg.predict(x_test))*100)+'%')
user_result = lg.predict(user_data)
st.subheader('Your Report: ')
output=''
if user_result[0]==0:
  output = 'You are not Diabetic.'
else:
  output = 'You are Diabetic.Consult a doctor.'

st.write(output)

