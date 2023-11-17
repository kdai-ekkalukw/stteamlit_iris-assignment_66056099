#66056099 Ekkaluk Wongsaman

import streamlit as st
import seaborn as sns
import matplotlib.pyplot as plt
import pandas as pd
import pickle
from sklearn.metrics import accuracy_score
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split

#deploy to streamlit cloud for 66056099 as URL below

st.title('Iris Classifier')
st.write("This app uses 6 inputs to predict the Iris using "
         "a model built on the Iris dataset. Use the form below"
         " to get started!")

iris_file = st.file_uploader('Upload your own iris data')

if iris_file is None:
    rf_pickle = open('random_forest_iris.pickle', 'rb')
    map_pickle = open('output_iris.pickle', 'rb')

    rfc = pickle.load(rf_pickle)
    unique_iris_mapping = pickle.load(map_pickle)

    rf_pickle.close()
else:
    iris_df = pd.read_csv(iris_file)
    penguin_df = iris_df.dropna()

    output = iris_df['species']
    features = iris_df[['sepal_length', 'sepal_width', 'petal_length',
                           'petal_width']]

    features = pd.get_dummies(features)

    output, unique_iris_mapping = pd.factorize(output)

    x_train, x_test, y_train, y_test = train_test_split(
        features, output, test_size=.8)

    rfc = RandomForestClassifier(random_state=15)
    rfc.fit(x_train, y_train)

    y_pred = rfc.predict(x_test)

    score = round(accuracy_score(y_pred, y_test), 2)

    st.write('We trained a Random Forest model on these data,'
             ' it has a score of {}! Use the '
             'inputs below to try out the model.'.format(score))

with st.form('user_inputs'):
    '''
    island = st.selectbox('', options=[
        'Biscoe', 'Dream', 'Torgerson'])
    sex = st.selectbox('Sex', options=[
        'Female', 'Male'])
    '''
    sepal_length = st.number_input(
        'Sepal Length', min_value=0, value=50)
    sepal_width = st.number_input(
        'Sepal Width', min_value=0, value=18)
    petal_length = st.number_input(
        'Petal Length', min_value=0, value=220)
    petal_width = st.number_input(
        'Petal Width', min_value=0, value=3650)
    st.form_submit_button()

'''
island_biscoe, island_dream, island_torgerson = 0, 0, 0
if island == 'Biscoe':
    island_biscoe = 1
elif island == 'Dream':
    island_dream = 1
elif island == 'Torgerson':
    island_torgerson = 1

sex_female, sex_male = 0, 0

if sex == 'Female':
    sex_female = 1

elif sex == 'Male':
    sex_male = 1

'''

new_prediction = rfc.predict([[sepal_length, sepal_width, petal_length,
                               petal_width]])
prediction_class = unique_iris_mapping[new_prediction][0]
st.write('We predict your iris is of the {} class'.format(prediction_class))
