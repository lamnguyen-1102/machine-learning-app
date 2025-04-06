import streamlit as st
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import warnings
from scipy import stats
import plotly.express as px

#from sklearn import preprocessing, impute
from sklearn.utils import resample
#from sklearn.feature_selection import SelectKBest, chi2
from sklearn.preprocessing import OneHotEncoder, StandardScaler, MinMaxScaler
#from sklearn.decomposition import KernelPCA, PCA
from sklearn.model_selection import StratifiedKFold, GridSearchCV, train_test_split, RandomizedSearchCV
from sklearn.metrics import accuracy_score, roc_auc_score, roc_curve, auc
from sklearn.metrics import f1_score, classification_report, confusion_matrix
# from sklearn.linear_model import LogisticRegression
# from sklearn.svm import SVC
# from sklearn.naive_bayes import GaussianNB
# from sklearn.neural_network import MLPClassifier
# from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
# import xgboost as xgb

warnings.filterwarnings("ignore")
random_state=42
np.random.seed(random_state)

#### Application Info ####
st.image("thumbnail2.PNG")

st.title('💻 Rainfall Prediction App 💻')

st.info("""
        Ever wondered if you should carry an umbrella tomorrow? \n 
        By training classification model on 10 years of daily weather observations in Australia, this app helps you predict wheather it will rain the next day ☔.
        """)

#### Read Data ####
df = pd.read_csv('Data/Raw/weatherAUS.csv')

with st.expander('Raw dataset', expanded=False):
    #st.subheader('Dataset Info')
    st.write("""
        The dataset contains daily weather observations from numerous Australian weather stations. \n
        The data is collected from 2008 to 2017 and contains 23 features, including information about wind speed, humidity, temperature, and rainfall. \n
        The Date column has been removed as it is not relevant for the prediction task. \n
        The target variable is 'RainTomorrow', which indicates whether it will rain the next day (Yes) or not (No). \n
        """)
    
    df = df.drop(columns=["Date"])

    st.write("**Explainatory variables:**")
    raw_features = df.drop(columns=['RainTomorrow']).head(5)
    raw_features

    st.write("**Target variable:**")
    raw_target = df[['RainTomorrow']].head(5)
    raw_target

#### Data quality checks ####
with st.expander('Exploratory data analysis', expanded=False):
    
    st.write("""
        The dataset contains 145460 rows and 23 columns. \n
        The data types of the features are a mix of integers, floats, and objects. \n
        There are missing values in several columns, particularly in 'Sunshine', 'Evaporation', 'Clould3pm', and 'Clould9am'. \n
        """) 
    
    st.write("**Data types:**")
    data_types = df.dtypes.to_frame().reset_index()
    data_types.columns = ['Feature', 'Data Type']
    data_types

    st.write("**Missing values:**")
    missing_values = df.isnull().sum().to_frame().reset_index()
    missing_values.columns = ['Feature', 'Missing Values']
    missing_values['Pct. Missing'] = np.round((missing_values['Missing Values']/df.shape[0]) * 100,1)
    missing_values = missing_values[missing_values['Missing Values'] > 0].sort_values(by='Missing Values', ascending=False)
    missing_values

    st.write("**Categorical features:**")
    categorical_features = df.select_dtypes(include=['object']).columns.tolist()
    categorical_features
    
    st.write("**Numerical features:**")
    numerical_features = df.select_dtypes(exclude=['object']).columns.tolist()
    numerical_features

    st.write("**Data range:**")
    data_range = df.describe()
    data_range


#### Data processing ####
with st.expander('Data processing', expanded=False):
    
    st.write("""
        Rows with missing values for RainToday and RainTomorrow have been removed. \n
        The missing values have been handled by filling them with the mean or mode of the respective columns by location. \n
        """)
    
    st.write("**Cleaned dataset:**")
    ## 1. Drop rows with missing values for RainToday and RainTomorrow
    df = df.dropna(axis=0, subset=['RainToday', 'RainTomorrow'])
    cleaned_df = df.copy()

    ## 2. Fill missing values for categorical features with mode
    # global mode of the columns
    global_mode_series = cleaned_df[categorical_features[1:-2]].mode(dropna=True)
    global_mode = global_mode_series.iloc[0] if not global_mode_series.empty else np.nan

    # location-wise mode
    location_mode = cleaned_df.groupby('Location')[categorical_features[1:-2]].transform(
        lambda x: x.mode(dropna=True).iloc[0] if not x.mode(dropna=True).empty else np.nan
    )

    # Fille missing values with global mode and location-wise mode
    cleaned_df[categorical_features[1:-2]] = cleaned_df[categorical_features[1:-2]].fillna(location_mode)
    cleaned_df[categorical_features[1:-2]] = cleaned_df[categorical_features[1:-2]].fillna(global_mode)

    ## 3. Fill missing values for numerical features with mean
    # global mean of the columns
    global_mode = cleaned_df[numerical_features].mean(skipna=True)

    # location-wise mean
    location_mode = cleaned_df.groupby('Location')[numerical_features].transform('mean')

    # Fille missing values with global mode and location-wise mode
    cleaned_df[numerical_features] = cleaned_df[numerical_features].fillna(location_mode)
    cleaned_df[numerical_features] = cleaned_df[numerical_features].fillna(global_mode)

    ## 4. Display cleaned dataset
    cleaned_df

    st.write("**Number of missing values in cleaned dataset:**")
    n = cleaned_df.isnull().sum().sum()
    n

#### Data visualization ####
with st.expander('Data visualization', expanded=False):
    ## Histogram of numerical features:
    st.write("**Histogram of numerical features:**")
    fig, axes = plt.subplots(len(numerical_features), 2, figsize=(12, len(numerical_features) * 5))
    for i, feature in enumerate(numerical_features):
        sns.histplot(df[feature], ax=axes[i, 0], kde=True)
        axes[i, 0].set_title(f'Distribution of {feature}')
        stats.probplot(df[feature], dist="norm", plot=axes[i, 1])
        axes[i, 1].set_title(f'Normal Q-Q Plot of {feature}')
    
    st.pyplot(fig)

    ## Correlation heatmap between numerical features
    st.write("**Correlation heatmap between numerical features:**")
    corri = cleaned_df[numerical_features].corr(method = 'spearman')
    mask = np.triu(np.ones_like(corri, dtype=bool))
    fig, ax = plt.subplots(figsize=(16,10))
    sns.heatmap(corri, ax = ax, annot=True, mask = mask)
    st.pyplot(fig)

    ## Pairplot of temperature and pressure features
    st.write("**Pairplot of temperature and pressure features:**")
    fig = sns.pairplot(data=cleaned_df, vars=('MaxTemp','MinTemp','Pressure9am','Pressure3pm'), hue='RainTomorrow')
    st.pyplot(fig)

    ## Location-wise average windspeed
    windspeed_df = cleaned_df.groupby(['Location'])[['WindSpeed9am', 'WindSpeed3pm']].mean()
    windspeed_df = windspeed_df.reset_index()

    fig = px.line(windspeed_df, x ='Location', y = windspeed_df.columns, title = 'Location-wise: Average WindSpeed', markers=True)
    st.plotly_chart(fig)

    ## Location-wise average humidity
    del windspeed_df
    humidity_df = cleaned_df.groupby(['Location'])[['Humidity9am', 'Humidity3pm']].mean()
    humidity_df = humidity_df.reset_index()

    fig = px.line(humidity_df, x ='Location', y = humidity_df.columns, title = 'Location-wise: Average Humidity', markers=True)
    st.plotly_chart(fig)

    ## Location-wise average pressure
    del humidity_df
    pressure_df = cleaned_df.groupby(['Location'])[['Pressure9am', 'Pressure3pm']].mean()
    pressure_df = pressure_df.reset_index()

    fig = px.line(pressure_df, x ='Location', y = pressure_df.columns, title = 'Location-wise: Average Pressure', markers=True)
    st.plotly_chart(fig)

    ## Location-wise temperature
    del pressure_df
    temperature_df = cleaned_df.groupby(['Location'])[['MinTemp', 'MaxTemp', 'Temp9am', 'Temp3pm']].mean()
    temperature_df = temperature_df.reset_index()

    fig = px.line(temperature_df, x ='Location', y = temperature_df.columns, title = 'Location-wise: Average Temperature', markers=True)
    del temperature_df

#### Feature engineering ####
with st.expander('Feature engineering', expanded=False):
    st.write("""
        Outliers are handled by capping at 25th and 75th percentile \n
        Categorical variables are encoded using one-hot encoding \n
        Imbalanced data is handled using upsampling of the minority class.
        """)
    
    ## 1. Cap outliers at 25th and 75th percentiles
    def cap_data(df, q1, q3):
        for col in df.columns:
            if (((df[col].dtype)=='float64') | ((df[col].dtype)=='int64')):
                percentiles = df[col].quantile([q1,q3]).values
                ###########################################################
                ############ Replace the outliers with lower & upper limits
                ###########################################################
                df[col][df[col] <= percentiles[0]] = percentiles[0]
                df[col][df[col] >= percentiles[1]] = percentiles[1]
            else:
                df[col]=df[col]
        return df
    
    def eda(dataframe, features):
        data_info = [{"Minimum": dataframe[col].min(), "Maximum": dataframe[col].max(), "Mode": dataframe[col].mode()[0]} for col in features]
        return pd.DataFrame(data_info, index = features).T

    cleaned_df = cap_data(cleaned_df, 0.25, 0.75)
    
    st.write("**Data range after capping outliers:**")
    data_range = eda(cleaned_df,numerical_features)
    data_range

    ## 2. Encode categorical variables using one-hot encoding
    encoder = OneHotEncoder(drop='first', sparse_output= False)
    encoded_features = encoder.fit_transform(cleaned_df[categorical_features[:-1]])
    encoded_features = pd.DataFrame(encoded_features, columns = encoder.get_feature_names_out(categorical_features[:-1]))


    st.write("**Encoded categorical features:**")
    encoded_features

    ## 3. Concatenate encoded features with cleaned_df
    encoded_df = pd.concat([cleaned_df[numerical_features], encoded_features, cleaned_df[categorical_features[-1:]]], axis = 1)
    encoded_df.dropna(inplace = True)

    ## 4. Handle imbalanced data using resampling
    majority_class_data = encoded_df[encoded_df['RainTomorrow'] == "No"]
    minority_class_data = encoded_df[encoded_df['RainTomorrow'] == "Yes"]

    upsampled_minority_class = resample(minority_class_data, replace = True, n_samples = len(majority_class_data))
    encoded_df = pd.concat([majority_class_data, upsampled_minority_class], axis = 0)

    st.write("**Value counts of target variable after resampling:**")
    value_counts = pd.DataFrame(encoded_df['RainTomorrow'].value_counts()).reset_index()
    value_counts

    del value_counts
    del upsampled_minority_class
    del majority_class_data
    del minority_class_data
    del encoded_features
    del data_range

#### Feature inputs ####
with st.sidebar:
    st.header("Input Parameters")
    Location = st.selectbox("Location",set(df['Location']))
    MinTemp = st.slider("MinTemp", float(df['MinTemp'].min()), float(df['MinTemp'].max()), float(df['MinTemp'].mean()), 0.1, key = "MinTemp")
    MaxTemp = st.slider("MaxTemp", float(df['MaxTemp'].min()), float(df['MaxTemp'].max()), float(df['MaxTemp'].mean()), 0.1, key = "MaxTemp")
    Rainfall = st.slider("Rainfall", float(df['Rainfall'].min()), float(df['Rainfall'].max()), float(df['Rainfall'].mean()), 0.1, key = "Rainfall")
    Evaporation = st.slider("Evaporation", float(df['Evaporation'].min()), float(df['Evaporation'].max()), float(df['Evaporation'].mean()), 0.1, key = "Evaporation")
    Sunshine = st.slider("Sunshine", float(df['Sunshine'].min()), float(df['Sunshine'].max()), float(df['Sunshine'].mean()), 0.1, key = "Sunshine")
    WindGustDir = st.selectbox("WindGustDir", set(df['WindGustDir']), key = "WindGustDir")
    WindGustSpeed = st.slider("WindGustSpeed", float(df['WindGustSpeed'].min()), float(df['WindGustSpeed'].max()), float(df['WindGustSpeed'].mean()), 0.1, key = "WindGustSpeed")
    WindDir9am = st.selectbox("WindDir9am", set(df['WindDir9am']), key = "WindDir9am")
    WindSpeed9am = st.slider("WindSpeed9am", float(df['WindSpeed9am'].min()), float(df['WindSpeed9am'].max()), float(df['WindSpeed9am'].mean()), 0.1, key = "WindSpeed9am")
    WindDir3pm = st.selectbox("WindDir3pm", set(df['WindDir3pm']), key = "WindDir3pm")
    WindSpeed3pm = st.slider("WindSpeed3pm", float(df['WindSpeed3pm'].min()), float(df['WindSpeed3pm'].max()), float(df['WindSpeed3pm'].mean()), 0.1, key = "WindSpeed3pm")
    Humidity9am = st.slider("Humidity9am", float(df['Humidity9am'].min()), float(df['Humidity9am'].max()), float(df['Humidity9am'].mean()), 1.0, key = "Humidity9am")
    Humidity3pm = st.slider("Humidity3pm", float(df['Humidity3pm'].min()), float(df['Humidity3pm'].max()), float(df['Humidity3pm'].mean()), 1.0, key = "Humidity3pm")
    Pressure9am = st.slider("Pressure9am", float(df['Pressure9am'].min()), float(df['Pressure9am'].max()), float(df['Pressure9am'].mean()), 0.1, key = "Pressure9am")
    Pressure3pm = st.slider("Pressure3pm", float(df['Pressure3pm'].min()), float(df['Pressure3pm'].max()), float(df['Pressure3pm'].mean()), 0.1, key = "Pressure3pm")
    Cloud9am = st.slider("Cloud9am", float(df['Cloud9am'].min()), float(df['Cloud9am'].max()), float(df['Cloud9am'].mean()), 1.0, key = "Cloud9am")
    Cloud3pm = st.slider("Cloud3pm", float(df['Cloud3pm'].min()), float(df['Cloud3pm'].max()), float(df['Cloud3pm'].mean()), 1.0, key = "Cloud3pm")
    Temp9am = st.slider("Temp9am", float(df['Temp9am'].min()), float(df['Temp9am'].max()), float(df['Temp9am'].mean()), 0.1, key = "Temp9am")
    Temp3pm = st.slider("Temp3pm", float(df['Temp3pm'].min()), float(df['Temp3pm'].max()), float(df['Temp3pm'].mean()), 0.1, key = "Temp3pm")
    RainToday = st.selectbox("RainToday", set(df['RainToday']), key = "RainToday")

    # Create dataframe for input parameters
    input_data = {
        'Location': Location,
        'MinTemp': MinTemp,
        'MaxTemp': MaxTemp,
        'Rainfall': Rainfall,
        'Evaporation': Evaporation,
        'Sunshine': Sunshine,
        'WindGustDir': WindGustDir,
        'WindGustSpeed': WindGustSpeed,
        'WindDir9am': WindDir9am,
        'WindSpeed9am': WindSpeed9am,
        'WindDir3pm': WindDir3pm,
        'WindSpeed3pm': WindSpeed3pm,
        'Humidity9am': Humidity9am,
        'Humidity3pm': Humidity3pm,
        'Pressure9am': Pressure9am,
        'Pressure3pm': Pressure3pm,
        'Cloud9am': Cloud9am,
        'Cloud3pm': Cloud3pm,
        'Temp9am': Temp9am,
        'Temp3pm': Temp3pm,
        'RainToday': RainToday
    }

    input_df = pd.DataFrame(input_data, index=[0])

#### Display and encode input parameters
with st.expander('Input parameters', expanded=False):
    st.write("**Input parameters:**")
    st.write(input_df)

    st.write("**Encoded input parameters:**")
    input_encoded = encoder.transform(input_df[categorical_features[:-1]])
    input_encoded = pd.DataFrame(input_encoded, columns = encoder.get_feature_names_out(categorical_features[:-1]))
    input_encoded = pd.concat([input_df[numerical_features], input_encoded], axis = 1)
    input_encoded

#### Model training
X = encoded_df.drop(columns=['RainTomorrow'])
y = encoded_df['RainTomorrow'].apply(lambda x: 1 if x == 'Yes' else 0)

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=random_state)
cv = StratifiedKFold(n_splits = 8, shuffle = True)

# Feature scaling
scaler = StandardScaler()
X_train[numerical_features] = scaler.fit_transform(X_train[numerical_features])
X_test[numerical_features] = scaler.transform(X_test[numerical_features])
input_encoded[numerical_features] = scaler.transform(input_encoded[numerical_features])

# Fitting models
def train_model_with_random_search(model, param_grid, X_train, y_train, X_test, y_test):
    # Initialize RandomizedSearchCV
    random_search = RandomizedSearchCV(model, param_distributions=param_grid, n_iter=5, 
                                       scoring='accuracy', n_jobs=-1)
    
    # Fit RandomizedSearchCV
    random_search.fit(X_train, y_train)
    
    # Get best estimator
    best_model = random_search.best_estimator_
    # Get best param
    best_param = random_search.best_params_
    # Predict on test set
    y_pred = best_model.predict(X_test)
    
    # Compute evaluation metrics
    accuracy = accuracy_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred, average='weighted')
    roc_auc = roc_auc_score(y_test, y_pred)
    
    print('Accuracy: ', accuracy)
    print('F1 Score: ', f1)
    print('AUC(ROC): ', roc_auc)
    print()
    print("Classification Report: ")
    print(classification_report(y_test, y_pred))
    
    ## ROC AUC
    prob = best_model.predict_proba(X_test)  
    prob = prob[:, 1]
    fper, tper, _ = roc_curve(y_test, prob)
    auc_scr = auc(fper, tper)
    
    
    fig, axes = plt.subplots(1, 2, figsize = (12, 6))
    axes[0].plot(fper, tper, color='darkorange', lw=2, label='ROC curve (area = %0.2f)' % auc_scr)
    axes[0].plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
    axes[0].set_xlabel('False Positive Rate')
    axes[0].set_ylabel('True Positive Rate')
    axes[0].set_title('Receiver Operating Characteristic (ROC) Curve')
    axes[0].legend(loc="lower right")
    
    sns.heatmap(confusion_matrix(y_test, y_pred), ax = axes[1], annot = True, cbar = False, fmt='.0f')
    axes[1].set_xlabel('Predicted labels')
    axes[1].set_ylabel('Actual labels')
    
    
    # Return evaluation metrics
    return model, accuracy, f1, roc_auc, best_param, fig

rf_param_grid = {
    'n_estimators': [400],
    'criterion': ['gini'],
    'max_depth': [50],
    'min_samples_split': [2],
    'min_samples_leaf': [1],
    'bootstrap': [True, False]
}

model_rf = RandomForestClassifier()
model_rf, acc_rf, f1_rf, roc_auc_rf, param_rf, fig_rf = train_model_with_random_search(model_rf, rf_param_grid, X_train, y_train, X_test, y_test)

with st.expander('Random Forest Classifier model evaluation', expanded=False):
    st.write("**Model evaluation metrics:**")
    st.write(f"Accuracy: {acc_rf:.4f}")
    st.write(f"F1 Score: {f1_rf:.4f}")
    st.write(f"AUC(ROC): {roc_auc_rf:.4f}")
    st.write(f"Best parameters: {param_rf}")

    st.pyplot(fig_rf)

st.subheader('Model prediction')

model_rf = RandomForestClassifier(**param_rf, random_state=random_state)
model_rf.fit(X,y)

y_pred = model_rf.predict(input_encoded)
y_pred_proba = model_rf.predict_proba(input_encoded)[:,1]

# Display prediction result
if y_pred[0] == 1:
    st.success(f"It will rain tomorrow! ☔, with a probability of {y_pred_proba[0]:.2f}%")
else:
    st.success(f"It will not rain tomorrow! 🌞, with a probability of {y_pred_proba[0]:.2f}%")