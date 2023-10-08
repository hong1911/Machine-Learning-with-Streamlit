import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import streamlit as st
import io
from utils import check_data, hist_plot, box_plot, heat_map, pair_grid
from utils import feat_eng, ML_model, select_target_var

st.set_page_config(layout="wide")
st.header('The End-to-End Machine Learning')

tab1, tab2, tab3, tab4, tab5 = st.tabs(['Import data','Clarify problem and establish metrics','Data Exploration','Data cleaning and feature engineering','Model training and evaluation'])

with tab1:
    exams = pd.read_csv('exams.csv')
    st.write(exams)

with tab2:
    st.subheader('CLARIFY THE PROBLEM AND CONSTRAINTS')
    st.write("""I aimed at providing valuable insights into student academic 
    performance across critical domains of mathematics, reading, and writing.
    Leveraging an extensive dataset encompassing demographic attributes such as 
    gender, race/ethnicity, parental level of education, and socio-economic 
    indicators including lunch programs and participation in test preparation
    courses, my project delves into the intricate web of factors shaping student
    success. Furthermore, I have developed an intuitive user interface (UI) for data
    visualization, allowing users to interact with and explore the data
    effortlessly. Additionally, I created a machine learning to predict
    student's math, reading, and writing score. Through rigorous analysis and
    predictive modeling, I strive to empower educational institutions with the
    knowledge and tools to optimize learning environments and support student on
    their path to excellence.
    """)

    st.subheader('ESTABLISH METRICS')
    st.write("""To assess the performance of student performance prediction model,
    I will employ several key regression metrics tailored including MAE
    (Mean Absolute Error), MSE (Mean Squared Error), and R2 to the nature of
    the task. These metrics will help us gauge the accuracy and precision of our
    predictions, as well as the overall goodness of fit of the model.
    """)
    
with tab3:
    eda_list = ['Data Overview', 'Visualization']
    col1,col2 = st.columns([2,8])
    EDA = col1.selectbox('Select EDA',eda_list)
    if EDA == 'Data Overview':
        col1,col2  = st.columns([1,3])
        with col1:
            st.write(f'Data shape: {exams.shape}')
            st.write('Data description:')
            st.dataframe(exams.describe())
        with col2:
            st.write("""Analysis:
                     Below table indicates that no data is missing,
                     one duplicated value. Several features are categorical
                     and I will convert them to dummies variables to
                     build regression model to predict math, reading, and writing score""")
            check_df = check_data(exams)
            st.dataframe(check_df)

    if EDA == 'Visualization':
        col1,col2 = st.columns([2,8])
        viz_list = ['Histogram','Box Plot','Heatmap and PairGrid']
        selected_viz = col1.selectbox('Select visualization',viz_list)
        if selected_viz == 'Histogram':
            col1,col2,col3,col4,col5 = st.columns(5)
            with col1:
                hist_plot('gender',exams)
            with col2:
                hist_plot('race/ethnicity',exams)
            with col3:
                hist_plot('parental level of education',exams)
            with col4:
                hist_plot('lunch',exams)
            with col5:
                hist_plot('test preparation course',exams)
        
        if selected_viz == 'Box Plot':
            col1, col2, col3, col4 = st.columns([1.2,2,2,2])
            feature_1_list = ['gender','race/ethnicity','parental level of education',
                            'lunch','test preparation course']
            with col1:
                feature_1 = st.radio('Select feature', feature_1_list)
            with col2:
                box_plot(feature_1, 'math score', exams)
            with col3:
                box_plot(feature_1, 'reading score', exams)
            with col4:
                box_plot(feature_1, 'writing score', exams)

        if selected_viz == 'Heatmap and PairGrid':
            col1, col2, col3, col4, col5 = st.columns([0.5, 1.25, 0.2, 0.6, 1.5])
            feature_hue = ['gender','race/ethnicity','parental level of education',
                            'lunch','test preparation course']
            with col1:
                width = st.slider('Select heatmap width', 1, 25, 16)
                height = st.slider('Select heatmap height', 1, 25, 6)
            with col2:
                heat_map(exams[['math score','reading score','writing score']], width, height)
            with col4:
                hue_ = st.selectbox('Select feature for PairGrid', feature_hue)
            with col5:
                pair_grid(exams, hue_ = hue_)
with tab4:
    col_list = ['gender','race/ethnicity','parental level of education',
            'lunch','test preparation course']
    clean_exams = feat_eng(exams,col_list)
    st.write("""Feature engineering: I removed duplicated values and convert category variables
             to dummies variables (drop the 1st level of each variable)""")
    st.write(f'Original data shape: {exams.shape}')
    st.write(f'Final data shape: {clean_exams.shape}')
    st.dataframe(clean_exams)

with tab5:
    st.subheader('Use the sidebar to select model, target variable, and other parameters')
    st.sidebar.write('The sidebar area is used to control the model only. It does not affect other tasks')
    # with st.sidebar.header('1. Choose the model'):
    selected_model = st.sidebar.selectbox('Select model',['Linear Regression','Random Forest','Lasso Regression'])
    selected_target_name = st.sidebar.selectbox('Select target variable',['Math', 'Reading', 'Writing'])
    selected_target_var = select_target_var(selected_target_name)
    clean_exams = feat_eng(exams,col_list)
    # ML_model(exams, selected_model, selected_target_var)
    if selected_model == 'Random Forest':
        # ML_model.parameter_layout()
        ML_model_instance = ML_model(clean_exams, selected_model, selected_target_var)
        ML_model_instance.run_rf_model()
    if selected_model == 'Linear Regression':
        ML_model_instance = ML_model(clean_exams, selected_model, selected_target_var)
        ML_model_instance.run_linear_model()
    if selected_model == 'Lasso Regression':
        ML_model_instance = ML_model(clean_exams, selected_model, selected_target_var)
        ML_model_instance.run_lasso_model()
