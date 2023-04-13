import streamlit as st
import pandas as pd
import geopandas as geopandas
import numpy as np
from folium import plugins, folium
import streamlit_folium as sf
from matplotlib import pyplot as plt
import seaborn as sns
from sklearn.preprocessing import OneHotEncoder
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression, Ridge, Lasso, BayesianRidge, ElasticNet
from sklearn.tree import DecisionTreeRegressor
from sklearn.metrics import explained_variance_score as evs
from sklearn.metrics import r2_score as r2

# Your code, refactored for Streamlit

# Importing dataset:
df = pd.read_json('preprocessed_old.json')

# Transform coordinates to geopandas
geometry = geopandas.points_from_xy(df.Lat, df.Lon)
geo_df = geopandas.GeoDataFrame(
    df[["Title", "Price", "Lon", "Lat"]], geometry=geometry
)
print(geo_df.head())

# Your existing code for creating the map
map = folium.Map(location=[47.036953, 28.833097], tiles="OpenStreetMap", zoom_start=7)

heat_data = [[point.xy[0][0], point.xy[1][0]] for point in geo_df.geometry]
plugins.HeatMap(heat_data, radius=12).add_to(map)

# Display the map in Streamlit
sf.folium_static(map)


df.drop(columns=['Title',
                 'Region',
                 'Author',
                 'AuthorProfile',
                 'Description',
                 'UpdatedAt',
                 'Lon',
                 'Type',
                 'Lat'], axis=1, inplace=True)

# OneHotEncoder for categorical features
s = (df.dtypes == 'object')
object_cols = list(s[s].index)
OH_encoder = OneHotEncoder(sparse_output=False)
OH_cols = pd.DataFrame(OH_encoder.fit_transform(df[object_cols]))
OH_cols.index = df.index
OH_cols.columns = OH_encoder.get_feature_names_out()
df_final = df.drop(object_cols, axis=1)
df_final = pd.concat([df_final, OH_cols], axis=1)

# Drop missing values
df_final.dropna(inplace=True)

# Display scatter plot for Price vs TotalArea
fig, ax = plt.subplots(figsize=(10, 6))
sns.scatterplot(x='TotalArea', y='Price', data=df_final, ax=ax)
ax.set_title('Price vs TotalArea')
st.pyplot(fig)

# Display box plot for Price distribution by NrRooms
fig, ax = plt.subplots(figsize=(10, 6))
sns.boxplot(x='NrRooms', y='Price', data=df_final, ax=ax)
ax.set_title('Price distribution by NrRooms')
st.pyplot(fig)


#%%
def remove_outliers_iqr(dataframe, columns, multiplier=1.5):
    for column in columns:
        Q1 = dataframe[column].quantile(0.25)
        Q3 = dataframe[column].quantile(0.75)
        IQR = Q3 - Q1

        lower_bound = Q1 - multiplier * IQR
        upper_bound = Q3 + multiplier * IQR

        dataframe = dataframe[(dataframe[column] >= lower_bound) & (dataframe[column] <= upper_bound)]

    return dataframe

# Remove outliers based on 'Price' and 'NrRooms' columns
df_no_outliers = remove_outliers_iqr(df_final, columns=['Price', 'NrRooms'])

# Display box plot for Price distribution by NrRooms after removing outliers
fig, ax = plt.subplots(figsize=(10, 6))
sns.boxplot(x='NrRooms', y='Price', data=df_no_outliers, ax=ax)
ax.set_title('Price distribution by NrRooms (after removing outliers)')
st.pyplot(fig)

# Train models and display evaluation scores as a bar plot

# Split dataset into training and testing
X_var = df_no_outliers[['TotalArea', 'NrRooms', 'Balcony', 'Floor', 'NumberOfFloors', 'HousingType_Construcţii noi', 'HousingType_Secundar', 'Condition_Are nevoie de reparație', 'Condition_Construcție nefinisată', 'Condition_Dat în exploatare', 'Condition_Design individual', 'Condition_Euroreparație', 'Condition_Fără reparație', 'Condition_La alb', 'Condition_Reparație cosmetică']].values
y_var = df_no_outliers['Price'].values

X_train, X_test, y_train, y_test = train_test_split(X_var, y_var, test_size=0.2, random_state=0)

#Models
#1. Linear Regression (OLS)
ols = LinearRegression()
ols.fit(X_train, y_train)
ols_yhat = ols.predict(X_test)

#2. Ridge Regression
ridge = Ridge(alpha=0.5)
ridge.fit(X_train, y_train)
ridge_yhat = ridge.predict(X_test)

# 3. Lasso Regression
lasso = Lasso(alpha=0.01)
lasso.fit(X_train, y_train)
lasso_yhat = lasso.predict(X_test)

# 4. Bayesian Ridge Regression
bayesian = BayesianRidge()
bayesian.fit(X_train, y_train)
bayesian_yhat = bayesian.predict(X_test)

# 5. ElasticNet Regression
en = ElasticNet(alpha=0.01)
en.fit(X_train, y_train)
en_yhat = en.predict(X_test)

# 6. Decision Tree Regressor
dtr = DecisionTreeRegressor(random_state=0)
dtr.fit(X_train, y_train)
dtr_yhat = dtr.predict(X_test)

# Calculate Explained Variance Score and R-squared for each model
evs_values = [evs(y_test, yhat) for yhat in [ols_yhat, ridge_yhat, lasso_yhat, bayesian_yhat, en_yhat, dtr_yhat]]
r2_values = [r2(y_test, yhat) for yhat in [ols_yhat, ridge_yhat, lasso_yhat, bayesian_yhat, en_yhat, dtr_yhat]]

# Define the labels and values for the bar plot
labels = ['OLS', 'Ridge', 'Lasso', 'Bayesian', 'ElasticNet', 'Decision Tree']
x = np.arange(len(labels)) # the label locations
width = 0.35 # the width of the bars

fig, ax = plt.subplots()
rects1 = ax.bar(x - width/2, evs_values, width, label='Explained Variance Score')
rects2 = ax.bar(x + width/2, r2_values, width, label='R-squared')

# Add some text for labels, title and custom x-axis tick labels, etc.
ax.set_ylabel('Scores')
ax.set_title('Explained Variance Score and R-squared for Different Models')
ax.set_xticks(x)
ax.set_xticklabels(labels)
ax.legend()

# Function to auto-label bars
def autolabel(rects):
 for rect in rects:
  height = rect.get_height()
  ax.annotate('%.2f' % height,
  xy=(rect.get_x() + rect.get_width() / 2, height),
  xytext=(0, 3), # 3 points vertical offset
  textcoords="offset points",
  ha='center', va='bottom')

autolabel(rects1)
autolabel(rects2)

st.pyplot(fig)


# Input widgets
st.header("Predict the price of an apartment")

total_area = st.number_input("Total Area (sq. meters)", value=50, min_value=10, max_value=500, step=1)
nr_rooms = st.number_input("Number of rooms", value=2, min_value=1, max_value=10, step=1)
balcony = st.number_input("Balcony", value=1, min_value=0, max_value=5, step=1)
floor = st.number_input("Floor", value=1, min_value=1, max_value=30, step=1)
number_of_floors = st.number_input("Number of floors in the building", value=5, min_value=1, max_value=50, step=1)

housing_type = st.selectbox("Housing Type", ['Construcţii noi', 'Secundar'])
condition = st.selectbox("Condition", ['Are nevoie de reparație', 'Construcție nefinisată', 'Dat în exploatare', 'Design individual', 'Euroreparație', 'Fără reparație', 'La alb', 'Reparație cosmetică'])

# Create a dataframe with user inputs
input_df = pd.DataFrame([[total_area, nr_rooms, balcony, floor, number_of_floors, housing_type, condition]], columns=['TotalArea', 'NrRooms', 'Balcony', 'Floor', 'NumberOfFloors', 'HousingType', 'Condition'])

# One-hot encoding for categorical inputs
input_OH_cols = pd.DataFrame(OH_encoder.transform(input_df[object_cols]))
input_OH_cols.index = input_df.index
input_OH_cols.columns = OH_encoder.get_feature_names_out()
input_df_final = input_df.drop(object_cols, axis=1)
input_df_final = pd.concat([input_df_final, input_OH_cols], axis=1)

# Make a prediction using the best model (replace `ols` with the best-performing model)
prediction = ols.predict(input_df_final)

# Display the prediction
st.header(f"Predicted Price: {prediction[0]:,.2f} EUR")