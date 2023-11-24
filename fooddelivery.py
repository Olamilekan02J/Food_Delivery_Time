import pandas as pd
import numpy as np
import streamlit as st
import warnings
warnings.filterwarnings('ignore')
import seaborn as sns
import matplotlib.pyplot as plt
import pickle
import tensorflow as tf
from sklearn.preprocessing import LabelEncoder
from datetime import datetime, timedelta

data = pd.read_csv('train.csv')
df = data.copy()

splitter = lambda x: x.split(' ')[1]
df['Time_taken'] = df['Time_taken(min)'].apply(splitter)
df.drop('Time_taken(min)',axis = 1, inplace = True)

categoricals = df.select_dtypes(include = ['object', 'category'])
numericals = df.select_dtypes(include = 'number')

for i in categoricals[['Delivery_person_Age', 'Delivery_person_Ratings', 'multiple_deliveries', 'Time_taken']].columns:
    df[i] = pd.to_numeric(df[i], errors = 'coerce')

cats = df.select_dtypes(include = ['object', 'category'])
nums = df.select_dtypes(include = 'number')

df.dropna(inplace = True)
df.drop(['ID','Delivery_person_ID'], axis = 1, inplace = True)

sel = df.copy()

def transformer(dataframe):
    from sklearn.preprocessing import StandardScaler, LabelEncoder
    scaler = StandardScaler()
    encoder = LabelEncoder()

    for i in dataframe.columns:
        if dataframe[i].dtypes != 'O':
            dataframe[i] = scaler.fit_transform(dataframe[[i]])
        else:
            dataframe[i] = encoder.fit_transform(dataframe[i])
    return dataframe
dx = df.copy()
df = transformer(df.drop('Time_taken',axis = 1))

sel_cols = ['Delivery_person_Age','Delivery_person_Ratings','Order_Date','Weatherconditions','Type_of_vehicle','Vehicle_condition','Type_of_order','Road_traffic_density','multiple_deliveries', 'Time_Order_picked', 'Time_Orderd','Delivery_location_latitude','Delivery_location_longitude','Restaurant_latitude','Restaurant_longitude']
df = df[sel_cols]

x = df
y = dx.Time_taken

from sklearn.model_selection import train_test_split
xtrain, xtest, ytrain, ytest = train_test_split(x, y, test_size = 0.20, random_state = 40, stratify = y)

# Modelling
from xgboost import XGBRegressor
from sklearn.metrics import r2_score

model = XGBRegressor()
model.fit(xtrain, ytrain)
cross_validation = model.predict(xtrain)
print(r2_score(cross_validation, ytrain))

cross_validation = model.predict(xtest)
print(f'XGBOOST MODEL TEST: {r2_score(cross_validation, ytest)}')


# save model
model = pickle.dump(model, open('Time_taken.pkl', 'wb'))
print('\nModel is saved\n')



# ..............STREAMLIT DEVELOPEMENT..........
model = pickle.load(open('Time_taken.pkl','rb'))

st.sidebar.image('pngwing.com (8).png', width = 300)
# Sidebar navigation
selected_page = st.sidebar.radio("Navigation", ["Home", "Modeling"])
def HomePage():
    # Streamlit app header
    st.markdown("<h1 style='text-align: center; color: #0F4C75;'>Time of Delivery Prediction</h1>", unsafe_allow_html=True)
    st.image('pngwing.com (7).png', width = 400)
    st.markdown("<p style='text-align: center; font-size: 16px;'>Revolutionizing logistics with precise delivery time estimates</p>", unsafe_allow_html=True)
    st.markdown("---")

    # Background story
    st.markdown("<h2 style='color: #0F4C75;'>Project Background</h2>", unsafe_allow_html=True)
    st.markdown("<p>In the bustling landscape of modern commerce, the demand for swift and accurate delivery services has become paramount. As consumers increasingly rely on the convenience of online shopping, businesses are challenged to optimize their logistics to meet these expectations...</p>", unsafe_allow_html=True)

    st.markdown("<h3 style='color: #0F4C75;'>The Need for Precision</h3>", unsafe_allow_html=True)
    st.markdown("<p>Imagine a world where waiting for a package is a thing of the past. In this era of instant gratification, customers expect not just timely deliveries but the exact moment their package will arrive...</p>", unsafe_allow_html=True)

    st.markdown("<h3 style='color: #0F4C75;'>Navigating Logistics Challenges</h3>", unsafe_allow_html=True)
    st.markdown("<p>The logistics landscape is riddled with challenges — traffic variations, unpredictable weather conditions, and dynamic delivery routes. Conventional delivery time estimates often fall short in accounting for these factors, resulting in frustrated customers and operational inefficiencies...</p>", unsafe_allow_html=True)

    st.markdown("<h3 style='color: #0F4C75;'>Machine Learning as the Navigator</h3>", unsafe_allow_html=True)
    st.markdown("<p>At the heart of our initiative is machine learning, a powerful tool that learns and adapts from historical data. By analyzing patterns in delivery times, considering external variables, and continuously learning from each delivery experience, our model becomes a navigator...</p>", unsafe_allow_html=True)

    st.markdown("<h3 style='color: #0F4C75;'>Benefits Beyond Accuracy</h3>", unsafe_allow_html=True)
    st.markdown("<p>Beyond the immediate benefits of precise delivery time estimates, our model aims to optimize delivery routes, reduce operational costs, and enhance overall customer satisfaction...</p>", unsafe_allow_html=True)

    st.markdown("<h3 style='color: #0F4C75;'>The Future of Delivery</h3>", unsafe_allow_html=True)
    st.markdown("<p>As we delve into this project, we envision a future where waiting for a delivery is a thing of the past, replaced by a seamless experience where customers know exactly when their package will arrive. Our time of delivery prediction model is not just a tool; it's a glimpse into the future of logistics...</p>", unsafe_allow_html=True)

    st.markdown("<p style='text-align: center;'>Join us on this exciting venture as we redefine the logistics landscape, one accurate delivery prediction at a time.</p>", unsafe_allow_html=True)
    st.markdown("---")

    # Streamlit app footer
    st.markdown("<p style='text-align: center; font-size: 12px;'>Created with ❤️ by JAYEOBA OLAMILEKAN</p>", unsafe_allow_html=True)

# Function to define the modeling page content
def modeling_page():
    st.markdown("<h1 style='text-align: center; color: #0F4C75;'>Modeling Time of Delivery</h1>", unsafe_allow_html=True)
    st.sidebar.markdown('<br><br><br>', unsafe_allow_html= True)
    st.write(sel.head())
    st.sidebar.image('del.png', width = 300,  caption = 'customer and deliver agent info')

    # Display content based on the selected page
if selected_page == "Home":
    HomePage()
elif selected_page == "Modeling":
    st.sidebar.markdown('<br>', unsafe_allow_html= True)
    modeling_page()

if selected_page == "Modeling":
    st.sidebar.markdown("Add your modeling content here")
    st.sidebar.markdown('<br>', unsafe_allow_html= True)
    min_date = datetime.now() - timedelta(days=150)
    Delivery_person_Age = st.sidebar.number_input("Delivery_person_Age",int(sel['Delivery_person_Age'].min()), int(sel['Delivery_person_Age'].max()))
    Delivery_person_Ratings = st.sidebar.slider("Delivery_person_Ratings",sel['Delivery_person_Ratings'].min(),sel['Delivery_person_Ratings'].max())
    Order_Date = st.sidebar.date_input("Order_Date",  min_value=min_date)
    Weatherconditions = st.sidebar.selectbox("Weatherconditions", sel['Weatherconditions'].unique())
    Type_of_vehicle = st.sidebar.selectbox("Type_of_vehicle", sel['Type_of_vehicle'].unique())
    Vehicle_condition = st.sidebar.selectbox("Vehicle_condition", sel['Vehicle_condition'].unique())
    Type_of_order = st.sidebar.selectbox("Type_of_order", sel['Type_of_order'].unique())
    Road_traffic_density = st.sidebar.selectbox("Road_traffic_density", sel['Road_traffic_density'].unique())
    multiple_deliveries = st.sidebar.selectbox("multiple_deliveries", sel['multiple_deliveries'].unique())
    Time_Orderd = st.sidebar.number_input('Time_Orderd')
    Time_Order_picked = st.sidebar.number_input('Time_Order_picked')
        # Add other input fields similarly
        # Assuming you have a DataFrame called 'sel' that contains your data
        # Extract the minimum and maximum values for latitude and longitude
    min_delivery_latitude = sel['Delivery_location_latitude'].min()
    max_delivery_latitude = sel['Delivery_location_latitude'].max()
    min_delivery_longitude = sel['Delivery_location_longitude'].min()
    max_delivery_longitude = sel['Delivery_location_longitude'].max()
    min_restaurant_latitude = sel['Restaurant_latitude'].min()
    max_restaurant_latitude = sel['Restaurant_latitude'].max()
    min_restaurant_longitude = sel['Restaurant_longitude'].min()
    max_restaurant_longitude = sel['Restaurant_longitude'].max()
        # Get latitude and longitude inputs from the user in the sidebar
    Delivery_location_latitude = st.sidebar.slider("Delivery Location Latitude",min_value=min_delivery_latitude,max_value=max_delivery_latitude,value=min_delivery_latitude,format="%.6f")
    Delivery_location_longitude = st.sidebar.slider("Delivery Location Longitude",min_value=min_delivery_longitude,max_value=max_delivery_longitude,value=min_delivery_longitude,format="%.6f")
    Restaurant_latitude = st.sidebar.slider("Restaurant Latitude",min_value=min_restaurant_latitude,max_value=max_restaurant_latitude,value=min_restaurant_latitude,format="%.6f")
    Restaurant_longitude = st.sidebar.slider("Restaurant Longitude", min_value=min_restaurant_longitude, max_value=max_restaurant_longitude, value=min_restaurant_longitude, format="%.6f")


      # Example: Create a dictionary from user input
    user_input = {
        'Delivery_person_Age': Delivery_person_Age,
        'Delivery_person_Ratings': Delivery_person_Ratings,
        'Order_Date': Order_Date,
        'Weatherconditions': Weatherconditions,
        'Type_of_vehicle': Type_of_vehicle,
        'Vehicle_condition': Vehicle_condition,
        'Type_of_order': Type_of_order,
        'Road_traffic_density': Road_traffic_density,
        'multiple_deliveries': multiple_deliveries,
        'Time_Order_picked': Time_Order_picked,
        'Time_Orderd': Time_Orderd,
        'Delivery_location_latitude': Delivery_location_latitude,
        'Delivery_location_longitude': Delivery_location_longitude,
        'Restaurant_latitude': Restaurant_latitude,
        'Restaurant_longitude': Restaurant_longitude,
    }

    # Create a DataFrame from the dictionary
    input_df = pd.DataFrame(user_input, index=[0])
    
    st.markdown("<h1 style='text-align: center; color: #0F4C75;'>USER INPUT</h1>", unsafe_allow_html=True)
    # Display the input DataFrame
    st.write(input_df)



        # Preprocess the input data
    categoricals = input_df.select_dtypes(include = ['object', 'category'])
    numericals = input_df.select_dtypes(include = 'number')
        
        # Standard Scale the Input Variable.
    from sklearn.preprocessing import StandardScaler, LabelEncoder
    scaler = StandardScaler()
    encoder = LabelEncoder()

    for i in numericals.columns:
        if i in input_df.columns:
            input_df[i] = scaler.fit_transform(input_df[[i]])
    for i in categoricals.columns:
        if i in input_df.columns: 
            input_df[i] = encoder.fit_transform(input_df[i])
    if st.button("Predict Delivery Time"):
        # Predict delivery time
        predicted_delivery_time = model.predict(input_df)
        formatted_time = "{:.2f}".format(predicted_delivery_time[0])
        st.success(f"The estimated delivery time is {formatted_time} minutes.")






# # # st.sidebar.image('user.png')




# # st.markdown('<hr>', unsafe_allow_html=True)
# # st.markdown("<h2 style = 'color: #0A2647; text-align: center; font-family: helvetica '>Model Report</h2>", unsafe_allow_html = True)









