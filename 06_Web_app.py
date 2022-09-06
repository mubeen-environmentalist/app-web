#import libraries
import streamlit as st
import numpy as np 
import matplotlib.pyplot as plt
from sklearn import datasets
from sklearn.model_selection import train_test_split
from sklearn.decomposition import PCA
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier 
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import accuracy_score

#App Heading
st.write('''
         # **Explore Different ML Models and Datasets**
         ''')

#Add a side bar
dataset_name = st.sidebar.selectbox(
    'select Dataset', 
    ('Iris', 'Breast Cancer', 'Wine')
)

#ML Models
classifier_name = st.sidebar.selectbox(
    'Select classifier',
    ('KNN', 'SVM', 'Random Forest')
)

#Import Dataset
def get_dataset(dataset_name):
    data = None
    if dataset_name == "Iris":
        data = datasets.load_iris()
    elif dataset_name == "Wine":
        data = datasets.load_wine()
    else:
        data = datasets.load_breast_cancer()
    X = data.data
    y = data.target
    return X, y

#Call the function and equal to  x, y
X,y = get_dataset(dataset_name)

#print shape of Data

st.write('Shape of Dataset: ', X.shape)
st.write('Number of Classes: ', len(np.unique(y)))

#Parameter of Classifiers
def add_parameter_ui (classifier_name):
    params = dict()       #Creat an Empty Dictionary
    if classifier_name == 'SVM':
        C = st.sidebar.slider('C', 0.01, 10.0)
        params['C'] = C        #Its the degree of correct Classifier
    elif classifier_name == 'KNN':
        K = st.sidebar.slider('K', 1, 15)
        params['K'] = K #     its the number of nearest neighbors
    else:
        max_depth = st.sidebar.slider('max_depth', 2, 15)
        params['max_depth'] = max_depth           # depth of tree that grow in random forest
        n_estimators = st.sidebar.slider('n_estimators', 1, 100)
        params['n_estimators'] = n_estimators              #number of trees
    return params
    
#Call the function that define and equal variab;es in params
params = add_parameter_ui(classifier_name)

#Make the classifier base on classifier_name and params
def get_classifier(classifier_name, params):
    clf = None
    if classifier_name == 'SVM':
        clf = SVC(C=params['C'])
    elif classifier_name == 'KNN':
        clf = KNeighborsClassifier(n_neighbors=params['K'])
    else:
        clf = clf = RandomForestRegressor(n_estimators=params['n_estimators'],
                                          max_depth=params['max_depth'], random_state=1234)
    return clf

#Call the function and equal clf variables
clf = get_classifier(classifier_name, params)

#Data Splitting
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=1234)

#Now fit our model
clf.fit(X_train, y_train)
y_pred = clf.predict(X_test)

#Accuracy Score
acc = accuracy_score(y_test, y_pred)
st.write(f'Classifier = {classifier_name}')
st.write(f'Accuracy =', acc)

#Scatter Plot
#two dimensinal plotting
pca = PCA(2)
X_projected = pca.fit_transform(X)

#Slicing of data in 0 and 1
x1 = X_projected[:, 0]
x2 = X_projected[:, 1]

fig = plt.figure()
plt.scatter(x1, x2,
            c=y, alpha=0.8,
            cmap='viridis')

plt.xlabel('Principle Component 1')
plt.ylabel('Principle Component 2')
plt.colorbar()

#Show
st.pyplot(fig)







