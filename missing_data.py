# Data Preprocessing

# Importing the libraries
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

# Importing the dataset
dataset = pd.read_csv('Data.csv')
X = dataset.iloc[:, :-1].values
Y = dataset.iloc[:, 3].values

# Taking care of missing data
from sklearn.impute  import SimpleImputer
imputer = SimpleImputer(missing_values =np.nan, strategy='mean')
imputer = imputer.fit(X[:, 1:3])
X[:, 1:3] = imputer.transform(X[:, 1:3])

# Convert Country  into numbers
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder
ct = ColumnTransformer(transformers= [('encoder', OneHotEncoder(), [0])], remainder = 'passthrough' )
X = np.array(ct.fit_transform(X))

# Convert Y into numbers
from sklearn.preprocessing import LabelEncoder
lc = LabelEncoder()
Y = lc.fit_transform(Y)


# Feature Scaling (Normalisation)
from sklearn.preprocessing import StandardScaler 
sc = StandardScaler()
X = sc.fit_transform(X) 

#Splitting datasets into Traing set and Test set
from sklearn.model_selection import train_test_split
X_train, X_test, Y_train, Y_test = train_test_split(X,Y, test_size = 0.2, random_state =0)


# print(X_train)
# print(Y_test)
# print(X_test)
# print(Y_train)


###Testing Streamlit output
import streamlit as st

st.title("Handling Missing Values")
st.tex(" 
Country	Age	Salary	Purchased
France	44	72000	No
Spain	27	48000	Yes
Germany	30	54000	No
Spain	38	61000	No
Germany	40		Yes
France	35	58000	Yes
Spain		52000	No
France	48	79000	Yes
Germany	50	83000	No
France	37	67000	Yes
")
# st.title("Feature Scaling")
st.text("Feature Scaling& Splitting datasets into Traing set and Test set")

st.title("X_train")
st.table(X_train)

st.title("Y_test")
st.table(Y_test)

st.title("X_test")
st.table(X_test)

st.title("Y_train")
st.table(Y_train)



st.title("Handling Missing Values")
chart_data = pd.DataFrame(
     np.random.randn(50, 4),
     columns=["X_train", "Y_test", "X_test", "Y_train"])

st.bar_chart(chart_data)






