import streamlit as st
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
from sklearn import model_selection
from PIL import Image

#set title
st.title('First ML APP')


def main():
    activities = ['EDA', 'Visualization', 'Model', 'About Us']
    options = st.sidebar.selectbox('Select Option:',activities)

    #uploading file
    data = st.file_uploader('Upload file:',type=['csv','xlsx','txt','json'])
    if data is not None:
        st.success('File uploaded succesfully')
        df = pd.read_csv(data)

        # Dealing with EDA part
        if options == 'EDA':
            st.header('Exploratory Data Analysis')    
            st.dataframe(df.head(50))

            if st.checkbox('Display Shape'):
                st.write(df.shape)

            if st.checkbox('Display Columns'):
                st.write(df.columns)

            if st.checkbox('Select Multiple Columns'):
                selected_columns = st.multiselect('Select Preferred Columns',df.columns)
                df1 = df[selected_columns]
                st.dataframe(df1)

            if st.checkbox('Display Summary'):
                st.write(df1.describe().T)

            if st.checkbox('Display Null Values'):
                st.write(df1.isnull().sum())

            if st.checkbox('Display Data Types'):
                st.write(df1.dtypes)

            if st.checkbox('Display Correlation of various data columns'):
                st.write(df1.corr(numeric_only=True))

        # Dealing with Visualization part

        elif options=='Visualization':
            st.header('Data Visualization')
            st.dataframe(df.head(50))

            if st.checkbox('Select Multiple Columns'):
                selected_col = st.multiselect('Select Multiple Columns',df.columns)
                df1 = df[selected_col]
                st.dataframe(df1)

            if st.checkbox('Display heatmap'):
                sns.heatmap(df1.corr(numeric_only=True), annot=True)
                st.pyplot(plt)

            if st.checkbox('Display Pairplot'):
                sns.pairplot(df1, diag_kind='kde')
                st.pyplot(plt)

            #for checking data imbalance you can use pie chart
            if st.checkbox('Pie chart(for Data Imbalance)'):
                all_columns = df.columns.to_list()
                pie_column = st.selectbox('Select your Target column',all_columns)
                piedata = df[pie_column].value_counts().plot.pie(autopct='%1.1f%%')
                st.write(piedata)
                # plt.pie(df[pie_column].value_counts(), autopct='%1.1f%%')
                st.pyplot(plt)

        # dealing with model building part
        elif options == 'Model':
            st.header("Model Building")
            st.dataframe(df.head(50))

            if st.checkbox('Select Multiple Columns'):
                selected_col = st.multiselect('Select Multiple Columns(Select target column at the end)',df.columns)
                df1 = df[selected_col]
                st.dataframe(df1)

                # dividing data in X and Y
                x = df1.iloc[:,:-1]
                # st.dataframe(x)
                y = df1.iloc[:,-1]
                # st.dataframe(y)

            seed  = st.sidebar.slider('Seed',1,200)
            classifier_name = st.sidebar.selectbox('Select the Classifier',('KNN', 'SVM', 'LR', 'DecisionTree', 'NaiveBayes'))

            def add_params(name_of_clf):
                params=dict()
                if name_of_clf == 'SVM':
                    c = st.sidebar.slider('C',0.01,15.0)
                    params['C']=c
                elif name_of_clf == 'KNN':
                    k= st.sidebar.slider('K',1,15)
                    params['K']=k
                return params
            
            # calling the function
            params = add_params(classifier_name)

            #getting our classifier
            def get_classifier(classifier, params):
                clf=None
                if classifier == 'KNN':
                    clf = KNeighborsClassifier(n_neighbors=params['K'])
                elif classifier == 'SVM':
                    clf = SVC(C=params['C'])
                elif classifier == 'LR':
                    clf = LogisticRegression()
                elif classifier == 'DecisionTree':
                    clf = DecisionTreeClassifier()
                elif classifier == 'NaiveBayes':
                    clf = GaussianNB()
                else:
                    st.warning('Select a classifier')

                return clf
            
            #calling the function
            clf = get_classifier(classifier_name, params)

            #splitting the data
            x_train,x_test,y_train,y_test = train_test_split(x,y, test_size=0.25, random_state=seed)

            #fitting the data
            clf.fit(x_train,y_train)

            # predicting the data
            yhat = clf.predict(x_test)
            # st.write('Predictions',yhat)

            #accuracy
            accuracy = accuracy_score(y_test, yhat)
            st.write('Classifier: ',classifier_name)
            st.write('Accuracy for model: ',accuracy)

    if options == 'About Us':
        st.markdown(' This is an interactive web page, feel free to use it')
        st.sidebar.write('Author: Bhavesh Kabdwal')
        st.balloons()


if __name__ == '__main__':
    main()

