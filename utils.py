from typing import Dict
from typing import List
from typing import Text
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import streamlit as st
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LinearRegression, Lasso
from sklearn.metrics import mean_squared_error, r2_score

def check_data(df:pd.DataFrame) -> pd.DataFrame:
    """
    Check number of unique values, duplicated values, missing values, data type
    Parameter
    ---------
    df: pandas dataframe
    """
    check_df = pd.DataFrame({'No. unique values':df.nunique(),
                             'No. duplicated values':df.duplicated().sum(),
                             'No. missing values':df.isnull().sum(),
                             'Data type':df.dtypes,
                             'Unique values': ''})
    check_df['Unique values'] = 0
    for index in check_df.index:
        unique_values = df[index].unique()
        unique_values_str = ', '.join(map(str, unique_values))  # Convert unique values to a comma-separated string
        check_df.loc[index, 'Unique values'] = unique_values_str

    return check_df

def hist_plot(column: Text, df:pd.DataFrame) -> None:
    fig = plt.figure(figsize=(3,3))
    sns.histplot(y=column, data=df, orientation='vertical', color='skyblue')
    plt.xlabel('Frequency')
    plt.title('Horizontal Histogram Count')
    st.pyplot(fig)

def box_plot(column_1: Text, column_2:Text, df:pd.DataFrame) -> None:
    fig = plt.figure(figsize=(2,2))
    sns.boxplot(y=column_1, x=column_2, data=df)
    plt.title('Box plot')
    st.pyplot(fig)

def heat_map(df:pd.DataFrame, width:float, height:float) -> None:
    # Compute the correlation matrix
    correlation_matrix = df.corr()
    fig = plt.figure(figsize=(width,height))
    # fig, ax = plt.subplots(figsize=(8, 6))
    sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm', square=True)
    plt.title('Correlation Heatmap of Test Scores')
    st.pyplot(fig)

st.cache_data
def pair_grid(df:pd.DataFrame,*, hue_:Text):
    g = sns.PairGrid(df, hue=hue_)
    g.map_diag(sns.histplot)
    g.map_upper(sns.scatterplot)
    g.map_lower(sns.kdeplot)
    g.add_legend()
    st.pyplot(g)

st.cache_data
def feat_eng(df:pd.DataFrame, col_list:List) -> pd.DataFrame:
    df = df.drop_duplicates()
    df_2 = pd.get_dummies(df, columns = col_list, drop_first=True)
    return df_2

def select_target_var(selected_target_name:Text) -> Text:
    target_name = ['Math','Reading','Writing']
    target_col = ['math score','reading score','writing score']
    target_dict = dict(zip(target_name,target_col))
    selected_target_var = target_dict.get(selected_target_name)
    return selected_target_var

class ML_model:
    def __init__(self, df, selected_model, selected_target_var):
        self.df = df
        self.selected_model = selected_model
        self.selected_target_var = selected_target_var

    def show_training_test_data_info(self, X_train, X_test, Y_train, Y_test, X, Y):
        st.subheader('1. Train and Test data summary')
        col1,col2 = st.columns(2)
        with col1:
            st.markdown('**1.1. Data split**')
            st.write('Training set')
            st.info(X_train.shape)
            st.write('Test set')
            st.info(X_test.shape)
        with col2:
            st.markdown('**1.2. Variable details**:')
            st.write('X variable')
            st.info(list(X.columns))
            st.write('Y variable')
            st.info(list(Y.columns))

    def model_performance(self, model_, X_train, Y_train, X_test, Y_test):
        st.subheader('2. Model Performance')
        col1, col2 = st.columns(2)
        with col1:
            st.markdown('**2.1. Training set**')
            Y_pred_train = model_.predict(X_train)
            st.write('Coefficient of determination ($R^2$):')
            st.info( r2_score(Y_train, Y_pred_train) )
            st.write('Error (MSE or MAE):')
            st.info( mean_squared_error(Y_train, Y_pred_train) )
        
        with col2:
            st.markdown('**2.2. Test set**')
            Y_pred_test = model_.predict(X_test)
            st.write('Coefficient of determination ($R^2$):')
            st.info( r2_score(Y_test, Y_pred_test) )
            st.write('Error (MSE or MAE):')
            st.info( mean_squared_error(Y_test, Y_pred_test) )

    def run_linear_model(self):
        with st.sidebar.header('Set Parameters'):
            split_percent = st.sidebar.slider('Data split ratio (% for Training Set)', 10, 90, 80, 5)

        X = self.df[[self.selected_target_var]]
        Y = self.df.loc[:, self.df.columns != self.selected_target_var]

        # Data splitting
        X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=(100-split_percent)/100)
        self.show_training_test_data_info(X_train, X_test, Y_train, Y_test, X, Y)

        regr = LinearRegression()
        regr.fit(X_train, Y_train)

        
        self.model_performance(regr,X_train,Y_train,X_test,Y_test)
    
    def run_lasso_model(self):
        with st.sidebar.header('Set Parameters'):
            split_percent = st.sidebar.slider('Data split ratio (% for Training Set)', 10, 90, 80, 5)

        X = self.df[[self.selected_target_var]]
        Y = self.df.loc[:, self.df.columns != self.selected_target_var]

        # Data splitting
        X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=(100-split_percent)/100)
        self.show_training_test_data_info(X_train, X_test, Y_train, Y_test, X, Y)

        lasso_regr = Lasso()
        lasso_regr.fit(X_train, Y_train)

        
        self.model_performance(lasso_regr,X_train,Y_train,X_test,Y_test)

    def run_rf_model(self):
        with st.sidebar.header('1. Set Parameters'):
            split_percent = st.sidebar.slider('Data split ratio (% for Training Set)', 10, 90, 80, 5)

        with st.sidebar.subheader('2. Learning Parameters'):
            parameter_n_estimators = st.sidebar.slider('Number of estimators (n_estimators)', 0, 1000, 100, 100)
            parameter_max_features = st.sidebar.selectbox('Max features (max_features)', options=['sqrt','log2'])
            parameter_min_samples_split = st.sidebar.slider('Minimum number of samples required to split an internal node (min_samples_split)', 1, 10, 2, 1)
            parameter_min_samples_leaf = st.sidebar.slider('Minimum number of samples required to be at a leaf node (min_samples_leaf)', 1, 10, 2, 1)

        with st.sidebar.subheader('2. General Parameters'):
            parameter_random_state = st.sidebar.slider('Seed number (random_state)', 0, 1000, 42, 1)
            parameter_criterion = st.sidebar.selectbox('Performance measure (criterion)', options=['squared_error', 'absolute_error', 'friedman_mse', 'poisson'])
            parameter_bootstrap = st.sidebar.select_slider('Bootstrap samples when building trees (bootstrap)', options=[True, False])
            parameter_oob_score = st.sidebar.select_slider('Whether to use out-of-bag samples to estimate the R^2 on unseen data (oob_score)', options=[False, True])

        X = self.df[[self.selected_target_var]]
        Y = self.df.loc[:,self.df.columns != self.selected_target_var]
        
        # Data splitting
        X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=(100-split_percent)/100)

        self.show_training_test_data_info(X_train, X_test, Y_train, Y_test, X, Y)

        rf = RandomForestRegressor(n_estimators=parameter_n_estimators,
            random_state=parameter_random_state,
            max_features=parameter_max_features,
            criterion=parameter_criterion,
            min_samples_split=parameter_min_samples_split,
            min_samples_leaf=parameter_min_samples_leaf,
            bootstrap=parameter_bootstrap,
            oob_score=parameter_oob_score)

        rf.fit(X_train, Y_train)

        st.subheader('2. Model Performance')
        col1, col2 = st.columns(2)
        with col1:
            st.markdown('**2.1. Training set**')
            Y_pred_train = rf.predict(X_train)
            st.write('Coefficient of determination ($R^2$):')
            st.info( r2_score(Y_train, Y_pred_train) )
            st.write('Error (MSE or MAE):')
            st.info( mean_squared_error(Y_train, Y_pred_train) )
        
        with col2:
            st.markdown('**2.2. Test set**')
            Y_pred_test = rf.predict(X_test)
            st.write('Coefficient of determination ($R^2$):')
            st.info( r2_score(Y_test, Y_pred_test) )
            st.write('Error (MSE or MAE):')
            st.info( mean_squared_error(Y_test, Y_pred_test) )

        st.subheader('3. Model Parameters')
        st.write(rf.get_params())
