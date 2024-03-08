import json
import time
import requests
import streamlit as st
from streamlit_lottie import st_lottie
from streamlit_lottie import st_lottie_spinner
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
import pandas as pd
from sklearn import tree
from sklearn.tree import DecisionTreeClassifier
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score,mean_squared_error
df = pd.read_csv("./data/iris.csv")

x=df.drop('variety',axis=1)
y=df['variety']
f=['sepal.length', 'sepal.width', 'petal.length', 'petal.width']
y.unique()
x_train,x_test,y_train,y_test =train_test_split(x,y,test_size=0.3,random_state=200)
man=DecisionTreeClassifier()
dt=man.fit(x_train,y_train)
yp=dt.predict(x_test)
s=accuracy_score(y_test,yp)
print(s)
xi=[[6.0,2.3,4.3,5]]
y_p=dt.predict(xi)
print(y_p)
y_p
'''streamlist'''

def load_lottieurl(url: str):
    r = requests.get(url)
    if r.status_code != 200:
        return None
    return r.json()

i0 = "https://lottie.host/2079e34e-be08-4813-9a05-7d3c6f8655a1/DSwgb6g2ol.json"
i01 = load_lottieurl(i0)
st.write("for predict")
s1=st.number_input("insert sepalwith")
s2=st.number_input("insert sepallengh")
s3=st.number_input("insert petalwith")
s4=st.number_input("insert petallengh")
if st.button("พยากรณ์ข้อมูล"):
    xi=[[s1,s2,s3,s4]]
    y_p=dt.predict(xi)
    st.success(y_p)
    st.button("ไม่พยากรณ์ข้อมูล")
else:
    st.button("ไม่พยากรณ์ข้อมูล")
st.write(f'accuracy{(s*100)}%')

st.title("Linear Regression")
st.header("by Ratthapong Somsao")
st_lottie(i01)
st.balloons()

st.header('Decision Tree For Classofocation')
st.write(df.head(10))
tree.plot_tree(dt)

fig, ax = plt.subplots(figsize=(12, 8))
tree.plot_tree(dt, feature_names=f, ax=ax)
st.pyplot(fig)
lottie.host
