from flask import Flask, request, render_template
import pickle
import numpy as np
import logging
import pymysql

app = Flask(__name__)

# Configure logging
logging.basicConfig(level=logging.DEBUG)

# Load the scaler and model
scaler_path = r'C:\Users\vishn\OneDrive\Desktop\employee churn\scaler.pkl'
model_path = r'C:\Users\vishn\OneDrive\Desktop\employee churn\xgb_model.pkl'

with open(scaler_path, 'rb') as scaler_file:
    scaler = pickle.load(scaler_file)

with open(model_path, 'rb') as model_file:
    model = pickle.load(model_file)


def get_db_connection():
    return pymysql.connect(host='localhost',
                           user='root',
                           password='viki@123',
                           db='emplo_churn',
                           charset='utf8mb4',
                           cursorclass=pymysql.cursors.DictCursor)

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    # Get data from form
    satisfaction_level = float(request.form['satisfaction_level'])
    last_evaluation = float(request.form['last_evaluation'])
    number_project = int(request.form['number_project'])
    average_montly_hours = int(request.form['average_montly_hours'])
    time_spend_company = int(request.form['time_spend_company'])
    work_accident = int(request.form['work_accident'])
    promotion_last_5years = int(request.form['promotion_last_5years'])
    salary = int(request.form['salary'])

    # Create a numpy array from the input data
    input_features = np.array([[satisfaction_level, last_evaluation, number_project, average_montly_hours, time_spend_company, work_accident, promotion_last_5years, salary]])
    
    # Log the input features before scaling
    logging.debug(f'Input features before scaling: {input_features}')

    # Scale the input features
    input_features_scaled = scaler.transform(input_features)

    # Log the input features after scaling
    logging.debug(f'Input features after scaling: {input_features_scaled}')

    # Predict using the model
    prediction = model.predict(input_features_scaled)

    # Log the prediction result
    logging.debug(f'Prediction result: {prediction}')

    # Render the result on the HTML page
    return render_template('index.html', prediction=int(prediction[0]))

@app.route('/employee_info', methods=['GET', 'POST'])
def employee_info():
    employee = None
    if request.method == 'POST':
        emp_id = request.form['emp_id']
        connection = get_db_connection()
        with connection.cursor() as cursor:
            sql = "SELECT * FROM dataset WHERE empid = %s"
            cursor.execute(sql, (emp_id,))
            employee = cursor.fetchone()
        connection.close()
    return render_template('employee_ret.html', employee=employee)

if __name__ == '__main__':
    app.run(debug=True)
