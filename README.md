# Rinex-Project-2-Weather_Data_Classification

# Decision Tree Classifier on Weather Data

This Python script applies a Decision Tree Classifier to predict whether the humidity level at 3 PM will exceed a threshold based on weather data collected at 9 AM. The model is trained using features such as air pressure, temperature, wind speed, and humidity recorded earlier in the day. The script also handles missing values and prepares the data for classification.

# Libraries Used

Pandas: For data manipulation and cleaning.

Scikit-learn: For machine learning tools, specifically the DecisionTreeClassifier and evaluation metrics.

NumPy: For numerical operations (indirectly used via Pandas).

# Steps in the Script

# 1.Loading and Exploring the Data:

python

Copy

data = pd.read_csv('daily_weather.csv')

data.columns

data.head()

The dataset daily_weather.csv is loaded into a Pandas DataFrame. The script explores the dataset by viewing the column names and the first few rows.

# 2.Handling Missing Values:

python

Copy

data[data.isnull().any(axis=1)].head()

del data['number']

The script checks for rows containing any missing values, and removes the column 'number' from the dataset.

# 3.Cleaning the Data:

python

Copy

data = data.dropna()

All rows with missing values are dropped to ensure the dataset is clean and ready for model training.

# 4.Feature Engineering:

python

Copy

clean_data['high_humidity_label'] = (clean_data['relative_humidity_3pm'] > 24.99) * 1

A new binary label high_humidity_label is created, where a value of 1 indicates that the humidity at 3 PM exceeds 24.99%, and 0 otherwise.

# 5.Splitting the Data:

python

Copy

morning_features = ['air_pressure_9am', 'air_temp_9am', 'avg_wind_direction_9am', 'avg_wind_speed_9am', 'max_wind_direction_9am', 'max_wind_speed_9am', 'rain_accumulation_9am', 'rain_duration_9am', 'relative_humidity_9am']

x = clean_data[morning_features].copy()

y = clean_data[['high_humidity_label']].copy()

The features (x) used to predict the target (y) are selected. The features include weather-related data recorded at 9 AM, while the target is the binary humidity label created earlier.

# 6.Train-Test Split:

python

Copy

X_train, X_test, y_train, y_test = train_test_split(x, y, test_size=0.33, random_state=324)

The data is split into training and testing sets (67% training, 33% testing) using train_test_split.

# 7.Model Training:

python

Copy

humidity_classifier = DecisionTreeClassifier(max_leaf_nodes=10, random_state=0)

humidity_classifier.fit(X_train, y_train)

A Decision Tree Classifier is instantiated with a maximum of 10 leaf nodes. The model is then trained on the training data.

# 8.Model Prediction:

python

Copy

y_predicted = humidity_classifier.predict(X_test)

The trained model is used to make predictions on the test data.

# 9.Evaluation:

python
Copy
accuracy_score(y_test, y_predicted) * 100

The accuracy of the model is calculated by comparing the predicted values (y_predicted) with the true values (y_test). The result is multiplied by 100 to get the accuracy as a percentage.

# Key Outputs

Accuracy:
The final accuracy score of the classifier is calculated and output as a percentage. This indicates how well the model predicts the occurrence of high humidity based on 9 AM weather data.

# Conclusion
This script demonstrates the use of a Decision Tree Classifier to predict whether the relative humidity at 3 PM will be high, based on weather conditions at 9 AM. The model is evaluated using accuracy, and data preprocessing steps ensure the dataset is clean and ready for analysis.

# Requirements

Python 3.x

Libraries: pandas, scikit-learn, numpy

You can install the necessary libraries using the following command:

bash

Copy

pip install pandas scikit-learn numpy

# Notes

The dataset file daily_weather.csv must be in the same directory as the script or provide the full path to the file.


