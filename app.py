import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

st.title("Iris")
st.markdown('สร้าง `scatter plot` แสดงผลข้อมูล **Iris**')

choices = ['sepal_length',
           'sepal_width',
           'petal_length',
           'petal_width']

# https://docs.streamlit.io/library/api-reference/widgets/st.selectbox
# 1. สร้าง st.selectbox ของ ตัวเลือก แกน x และ y จาก choices
#selected_x_var = 'อะไรดี'
#selected_y_var = 'อะไรดี'
selected_x_var = st.selectbox('select x variable',choices)
selected_y_var = st.selectbox('select y variable', choices)

# https://docs.streamlit.io/library/api-reference/widgets/st.file_uploader
# 2. สร้าง st.file_uploader เพื่อให้เลือกไฟล์ .csv เท่านั้น จากเครื่องผู้ใช้งาน
#penguin_file = None
iris_file = st.file_uploader('select file iris.csv then upload', type=['csv'])

if iris_file is not None:
    iris_df = pd.read_csv(iris_file)
else:
    st.stop()

st.subheader('ข้อมูลตัวอย่าง')
# st.write(penguins_df)

st.subheader('แสดงผลข้อมูล')
sns.set_style('darkgrid')
markers = {"Virginica": "v", "Versicolor": "s", "Setosa": 'o'}

fig, ax = plt.subplots()
ax = sns.scatterplot(data=iris_df,
                     x=selected_x_var, y=selected_y_var,
                     hue='species', markers=markers, style='species')
plt.xlabel(selected_x_var)
plt.ylabel(selected_y_var)
plt.title("Iris Data")
st.pyplot(fig)
