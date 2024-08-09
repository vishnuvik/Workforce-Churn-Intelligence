from flask import Flask, render_template, request
import pickle
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import load_model

app = Flask(__name__)

# Load the pre-trained models and encoders
company_binary_encoder = pickle.load(open('C:/Users/vishn/OneDrive/Desktop/Job market prediction/company_binary_encoder.pkl', 'rb'))
label_encoders = pickle.load(open('C:/Users/vishn/OneDrive/Desktop/Job market prediction/label_encoders.pkl', 'rb'))
minmax_scaler = pickle.load(open('C:/Users/vishn/OneDrive/Desktop/Job market prediction/MinMaxScaler.pkl', 'rb'))
salary_scaler = pickle.load(open('C:/Users/vishn/OneDrive/Desktop/Job market prediction/Salary_scale.pkl', 'rb'))
model = load_model('C:/Users/vishn/OneDrive/Desktop/Job market prediction/best_model.h5')

# Define high and low skills sets (all lowercase)
high_skills_set = {'tableau', 'powerpoint', 'vba', 'office', 'phd', 'machine learning', 'mba', 'docker', 'sap', 'spark', 'master', 'dynamics 365', 'sql', 'agile', 'python', 'jira', 'snowflake', 'erp', 'excel', 'hadoop', 'javascript', 'azure', 'tensor flow', 'word', 'databricks', 'deep learning', 'access', 'artificial intelligence', 'power bi', 'oracle', 'teradata', 'aws', 'c++', 'cpa', 'r', 'bachelor', 'java', 'english', 'google cloud'}
low_skills_set = {'github', 'looker', 'css', 'mongodb', 'pandas', 'dax', 'hyperion', 'spanish', 'seaborn', 'scikit', 'google sheets', 'php', 'matlab', 'c#', 'power pivot', 'french', 'd3', 'polars', 'power automate', 'matplotlib', 'numpy', 'sage', 'plotly', 'russian', 'qlik', 'angular', 'rust', 'power query', 'neural network', 'fabric', 'essbase', 'dash', 'ssis', 'sap analytics cloud', '.net', 'japanese', 'navision', 'salesforce', 'german', 'quickbooks', 'abap', 'react', 'html', 'snaplogic', 'airflow', 'cma', 'cfa', 'adobe analytics', 'kaggle', 'streamlit', 'chat gpt', 'jupyter', 'ssrs', 'chinese', 'power apps', 'cognos', 'domo', 'ssas'}

@app.route('/')
def home():
    return render_template('home.html')

@app.route('/predict', methods=['GET', 'POST'])
def predict():
    if request.method == 'POST':
        job_title = request.form['job_title']
        position = request.form['position']
        work_location = request.form['work_location']
        company_name = request.form['company_name']
        state = request.form['state']
        skills = request.form.getlist('skills[]')
        tenure = int(request.form['tenure'])
        experience = int(request.form['experience'])

        # Convert skills to lowercase
        skills = [skill.lower() for skill in skills]

        # Process skills input
        high_skills_count = sum(1 for skill in skills if skill in high_skills_set)
        low_skills_count = sum(1 for skill in skills if skill in low_skills_set)
        
        # Ensure the counts match the number of skills provided
        if high_skills_count + low_skills_count != len(skills):
            return render_template('pred.html', error="Some skills are not recognized.")

        # Process the inputs
        company_encoded = company_binary_encoder.transform([[company_name]])
        job_title_encoded = label_encoders['job_title'].transform([job_title])
        position_encoded = label_encoders['position'].transform([position])
        work_location_encoded = label_encoders['work_location'].transform([work_location])
        state_encoded = label_encoders['state'].transform([state])

        # Combine all encoded features into a single array
        combined_encoded_features = np.hstack([
            job_title_encoded,
            position_encoded,
            work_location_encoded,
            company_encoded,
            state_encoded
        ])

        # Scale all encoded features
        scaled_features = minmax_scaler.transform(combined_encoded_features)

        # Combine scaled features with high and low skills count, tenure, and experience
        features = np.hstack([
            scaled_features.flatten(),
            [high_skills_count],
            [low_skills_count],
            minmax_scaler.transform([[tenure]])[0],
            minmax_scaler.transform([[experience]])[0]
        ])

        # Make prediction
        features = features.reshape(1, -1)  # Reshape for prediction if needed
        prediction = model.predict(features)
        salary = salary_scaler.inverse_transform(prediction)[0][0]

        return render_template('pred.html', prediction=salary)
    return render_template('pred.html')

if __name__ == '__main__':
    app.run(debug=True)
