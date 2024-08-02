from flask import Flask, request, render_template
import pandas as pd
import numpy as np
import pickle
from sklearn.model_selection import train_test_split, RandomizedSearchCV
from sklearn.preprocessing import LabelEncoder
from sklearn.ensemble import RandomForestClassifier

app = Flask(__name__)

# Load and preprocess data
df = pd.read_csv("C:/Users/mails/Downloads/CRS_FYP/CRS_FYP/Crop_recommendation.csv")
class_labels = df['label'].unique().tolist()
le = LabelEncoder()
df['label'] = le.fit_transform(df['label'])
class_labels = le.classes_

# Split the data
x = df.drop('label', axis=1)
y = df['label']
features_data = {'columns': list(x.columns)}

# Train the model
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.10, shuffle=True)
rf_model = RandomForestClassifier()
rf_model.fit(x_train, y_train)

# Hyperparameter tuning
param_grid = {
    'n_estimators': np.arange(50, 200),
    'criterion': ['gini', 'entropy'],
    'max_depth': np.arange(2, 25),
    'min_samples_split': np.arange(2, 25),
    'min_samples_leaf': np.arange(2, 25)
}
rscv_model = RandomizedSearchCV(rf_model, param_grid, cv=5)
rscv_model.fit(x_train, y_train)
new_rf_model = rscv_model.best_estimator_

# Save the model
with open('new_rf_model.pickle', 'wb') as file:
    pickle.dump(new_rf_model, file)

# Save the column names
with open('features_data.pickle', 'wb') as file:
    pickle.dump(features_data, file)

@app.route('/')
def home():
    return render_template('home.html')

@app.route('/recommendation')
def recommendation():
    return render_template('recommendation.html')

@app.route('/predict', methods=['POST'])
def predict():
    # Load the model and features_data
    with open('new_rf_model.pickle', 'rb') as file:
        model = pickle.load(file)
    with open('features_data.pickle', 'rb') as file:
        features_data = pickle.load(file)
        
    # Get the input values from the form
    N = float(request.form['N'])
    P = float(request.form['P'])
    K = float(request.form['K'])
    temperature = float(request.form['temperature'])
    humidity = float(request.form['humidity'])
    ph = float(request.form['ph'])
    rainfall = float(request.form['rainfall'])
    
    # Create a DataFrame from the input values
    input_df = pd.DataFrame([[N, P, K, temperature, humidity, ph, rainfall]], columns=features_data['columns'])
    
    # Make prediction
    crop_index = model.predict(input_df)[0]
    recommended_crop = class_labels[crop_index]
    
    return render_template('recommendation.html', recommended_crop=recommended_crop)

if __name__ == '__main__':
    app.run(debug=True)
