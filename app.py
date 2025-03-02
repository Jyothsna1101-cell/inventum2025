import requests

url = "https://api.openweathermap.org/data/2.5/weather?q=London&appid=3b0cebc5ddb9ea2b0c1c026810308f84"

response = requests.get(url)

if response.status_code == 200: 
    data = response.json() 
    print(data)
else:
    print(f"Error: {response.status_code}")
    
import pandas as pd # Extract relevant fields
weather_data = { 
"coord": data["coord"],
"weather": data["weather"],
"base": data["base"],
"main": data["main"],
"visibility": data["visibility"],
"wind": data["wind"],
"clouds": data["clouds"],
"dt": data["dt"],
"sys": data["sys"],
"timezone": data["timezone"],
"id": data["id"],
"name": data["name"],
"cod": data["cod"],
}
# Convert to Pandas DataFrame
df = pd.DataFrame([weather_data])
print(df)

import matplotlib.pyplot as plt

# Assuming 'data' is the dictionary containing the weather data.
data = {
    "weather": [{"id": 801, "main": "Clouds", "description": "few clouds", "icon": "02d"}],
    "wind": {"speed": 5.14, "deg": 240},
    "clouds": {"all": 27},
    "visibility": 10000
}

# Extract the relevant numerical data for the plot:
weather_main = len(data["weather"])  # Number of weather conditions
wind_speed = data["wind"]["speed"]  # Wind speed
cloud_coverage = data["clouds"]["all"]  # Cloud coverage percentage
visibility = data["visibility"]  # Visibility in meters

# Labels and values for the bar plot
labels = ["Weather", "Wind Speed", "Cloud Coverage", "Visibility"]
values = [weather_main, wind_speed, cloud_coverage, visibility]

# Create the bar plot
plt.figure(figsize=(8,5))
plt.bar(labels, values, color=['blue', 'orange', 'green', 'red'])
plt.xlabel("Category")
plt.ylabel("Value")
plt.title("Weather Data")
plt.show()

import pandas as pd
import numpy as np

# Generate random historical weather data for the past 30 days
np.random.seed(42)

# Simulate weather data for 30 days
temperature = np.random.uniform(15, 35, size=30)  # Temperature in Celsius (between 15 and 35)
wind_speed = np.random.uniform(3, 15, size=30)  # Wind speed in m/s (between 3 and 15 m/s)
cloud_coverage = np.random.randint(0, 101, size=30)  # Cloud coverage in percentage (0-100%)
visibility = np.random.randint(5000, 10000, size=30)  # Visibility in meters (between 5000 and 10000 meters)

# Create DataFrame
df_weather = pd.DataFrame({
    "temperature": temperature,
    "wind_speed": wind_speed,
    "cloud_coverage": cloud_coverage,
    "visibility": visibility
})

# Add "day" column representing the past 30 days
df_weather["day"] = range(1, 31)

# Print the first few rows of the generated data
print(df_weather.head())

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression

# Generate random historical weather data for the past 30 days
np.random.seed(42)

# Simulate weather data for 30 days
temperature = np.random.uniform(15, 35, size=30)  # Temperature in Celsius (between 15 and 35)
wind_speed = np.random.uniform(3, 15, size=30)  # Wind speed in m/s (between 3 and 15 m/s)
cloud_coverage = np.random.randint(0, 101, size=30)  # Cloud coverage in percentage (0-100%)
visibility = np.random.randint(5000, 10000, size=30)  # Visibility in meters (between 5000 and 10000 meters)

# Create DataFrame
df_weather = pd.DataFrame({
    "temperature": temperature,
    "wind_speed": wind_speed,
    "cloud_coverage": cloud_coverage,
    "visibility": visibility
})

# Add "day" column representing the past 30 days
df_weather["day"] = range(1, 31)

# Features (X) and target (y)
X = df_weather[["day"]]  # Using "day" as the feature (independent variable)
y = df_weather["temperature"]  # Using "temperature" as the target (dependent variable)
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression

# Generate random historical weather data for the past 30 days
np.random.seed(42)

# Simulate weather data for 30 days
temperature = np.random.uniform(15, 35, size=30)  # Temperature in Celsius (between 15 and 35)
wind_speed = np.random.uniform(3, 15, size=30)  # Wind speed in m/s (between 3 and 15 m/s)
cloud_coverage = np.random.randint(0, 101, size=30)  # Cloud coverage in percentage (0-100%)
visibility = np.random.randint(5000, 10000, size=30)  # Visibility in meters (between 5000 and 10000 meters)

# Create DataFrame
df_weather = pd.DataFrame({
    "temperature": temperature,
    "wind_speed": wind_speed,
    "cloud_coverage": cloud_coverage,
    "visibility": visibility
})

# Add "day" column representing the past 30 days
df_weather["day"] = range(1, 31)

# Features (X) and target (y)
X = df_weather[["day"]]  # Using "day" as the feature (independent variable)
y = df_weather["temperature"]  # Using "temperature" as the target (dependent variable)

import streamlit as st
import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split

# Generate random historical weather data for the past 30 days
np.random.seed(42)

# Simulate weather data for 30 days
temperature = np.random.uniform(15, 35, size=30)  # Temperature in Celsius (between 15 and 35)
wind_speed = np.random.uniform(3, 15, size=30)  # Wind speed in m/s (between 3 and 15 m/s)
cloud_coverage = np.random.randint(0, 101, size=30)  # Cloud coverage in percentage (0-100%)
visibility = np.random.randint(5000, 10000, size=30)  # Visibility in meters (between 5000 and 10000 meters)

# Create DataFrame
df_weather = pd.DataFrame({
    "temperature": temperature,
    "wind_speed": wind_speed,
    "cloud_coverage": cloud_coverage,
    "visibility": visibility
})

# Add "day" column representing the past 30 days
df_weather["day"] = range(1, 31)

# Features (X) and target (y)
X = df_weather[["day"]]  # Using "day" as the feature (independent variable)
y = df_weather["temperature"]  # Using "temperature" as the target (dependent variable)

# Split data into training and testing sets (80% train, 20% test)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Create and train the Linear Regression model
model = LinearRegression()
model.fit(X_train, y_train)

# Streamlit App
st.title("Weather Prediction")
st.write("Predicting temperature for the next day based on historical weather data.")

# User Input
day_input = st.number_input("Enter day number (e.g., 31 for prediction)", min_value=1, max_value=100)

if st.button("Predict"):
    # Predict temperature for the entered day
    prediction = model.predict([[day_input]])
    
    # Show the result
    st.write(f"Predicted temperature for Day {day_input}: {prediction[0]:.2f}°C")
