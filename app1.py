from flask import Flask, render_template, request, jsonify
import pandas as pd
import numpy as np
import joblib
from werkzeug.security import generate_password_hash, check_password_hash

app = Flask(__name__)
from flask_sqlalchemy import SQLAlchemy
from flask import session, redirect, url_for, flash

app.secret_key = 'your_secret_key'  # Required for sessions
app.config['SQLALCHEMY_DATABASE_URI'] = 'sqlite:///users.db'
db = SQLAlchemy(app)

# Define User model
class User(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    username = db.Column(db.String(150), unique=True, nullable=False)
    password = db.Column(db.String(150), nullable=False)

# Load the dataset and preprocess
df = pd.read_csv('2020.csv')  # Adjust path if needed

# Load the model, encoders, and scaler
model = joblib.load('admission_prediction_model.pkl')
encoders = joblib.load('encoders.pkl')
scaler1 = joblib.load('scaler1.pkl')

# Extract unique values for the dropdown columns
unique_columns = {
    'Academic_Program': df['Academic Program Name'].unique().tolist(),
    'Quota': df['Quota'].unique().tolist(),
    'Seat_Type': df['Seat Type'].unique().tolist(),
    'Gender': df['Gender'].unique().tolist()
}

@app.route('/')
def home():
    username = session.get('user')
    return render_template('index1.html', unique_columns=unique_columns, username=username)


@app.route('/predict', methods=['POST'])
def predict():
    # Collect form data
    selected_values = {
        'Academic_Program': request.form.get('academic_program'),
        'Quota': request.form.get('quota'),
        'Seat_Type': request.form.get('seat_type'),
        'Gender': request.form.get('gender'),
        'Opening_Rank': float(request.form.get('opening_rank')),
        'Closing_Rank': float(4704),
        'Year': int(request.form.get('year')),
        'Round': int(request.form.get('round'))
    }

    # Convert input to DataFrame
    input_df = pd.DataFrame([selected_values])

    # Rename columns to match the model's expected column names
    input_df.rename(columns={
        'Academic_Program': 'Academic Program Name',
        'Opening_Rank': 'Opening Rank',
        'Closing_Rank': 'Closing Rank',
        'Seat_Type': 'Seat Type',
        'Quota': 'Quota',  # Ensure Quota is also properly matched
        'Gender': 'Gender'
    }, inplace=True)

    # Encode categorical values
    for col, encoder in encoders.items():
        if col in input_df:
            input_df[col] = encoder.transform(input_df[col])

    # Scale input
    input_scaled = scaler1.transform(input_df)

    # Predict probabilities for all institutes
    probabilities = model.predict_proba(input_scaled)

    # Get top 10 predicted colleges based on probabilities
    top_10_indices = np.argsort(probabilities, axis=1)[:, -10:]
    top_10_colleges = encoders['Institute'].inverse_transform(top_10_indices.flatten())
    result = "You have a high chance of admission!"  # example
    username = session.get('username')
    return render_template('result2.html', selected_values=selected_values, predicted_colleges=top_10_colleges.tolist(),prediction_result=result,username=username)
@app.route('/login', methods=['GET', 'POST'])
def login():
    if request.method == 'POST':
        user = User.query.filter_by(username=request.form['username']).first()
        if user and check_password_hash(user.password, request.form['password']):
            session['user'] = user.username
            flash('Login successful!', 'success')
            return redirect(url_for('home'))
        else:
            flash('Invalid credentials', 'danger')
    return render_template('login.html')


@app.route('/signup', methods=['GET', 'POST'])
def signup():
    if request.method == 'POST':
        existing_user = User.query.filter_by(username=request.form['username']).first()
        if existing_user:
            flash('User already exists', 'warning')
        else:
            new_user = User(username=request.form['username'],
                password=generate_password_hash(request.form['password']))
            db.session.add(new_user)
            db.session.commit()
            flash('Signup successful!', 'success')
            return redirect(url_for('login'))
    return render_template('signup.html')


@app.route('/logout')
def logout():
    session.pop('user', None)
    flash('Logged out successfully.', 'info')
    return render_template('logout.html')


with app.app_context():
    db.create_all()

if __name__ == '__main__':
    app.run(debug=True)
