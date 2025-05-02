from flask import Flask, render_template, request, redirect, url_for, session, flash
import pandas as pd
import matplotlib.pyplot as plt
import io
import base64
import joblib
import warnings
from sklearn.exceptions import InconsistentVersionWarning

# Suppress version mismatch warnings
warnings.filterwarnings("ignore", category=InconsistentVersionWarning)

app = Flask(__name__)
app.secret_key = "123"

# Load model and vectorizer
vectorizer = joblib.load('vectorizer.pkl')
model = joblib.load('support_vector.pkl')

# Dataset path
dataset_path = 'hazard.csv'

@app.route("/", methods=["GET", "POST"])
def index():
    if request.method == "POST" and 'submit' in request.form:
        uname = request.form.get("uname")
        pswd = request.form.get("pswd")

        if uname == "admin" and pswd == "admin":
            return redirect(url_for('result'))
        else:
            flash('Invalid Authentication', 'error')
            
    return render_template('index.html')

@app.route("/signout")
def signout():
    return render_template('index.html')

@app.route('/result')
def result():
    df = pd.read_csv(dataset_path)
    df = df.replace(r'\n', ' ', regex=True)
    columns_to_display = ['url', 'type']  # Show only relevant columns
    return render_template('result.html', columns=columns_to_display, rows=df.to_dict(orient='records'))

@app.route('/prediction', methods=['GET', 'POST'])
def predict():
    result = None
    input_text = None
    if request.method == 'POST':
        input_text = request.form['input_text']
        input_features = vectorizer.transform([input_text])
        prediction = model.predict(input_features)[0]

        label_map = {
            0: "phishing",
            1: "benign",
            2: "defacement",
            3: "malware"
        }
        result = label_map.get(prediction, "Unknown")
        flash(result)
        
    return render_template('prediction.html', input_text=input_text, result=result)

@app.route('/charts')
def charts():
    df = pd.read_csv(dataset_path)
    label_counts = df['label'].value_counts()

    plt.figure(figsize=(6, 4))
    plt.bar(label_counts.index, label_counts.values, color='skyblue')
    plt.xlabel('Labels')
    plt.ylabel('Counts')
    plt.title('Dataset Label Distribution')
    
    buf = io.BytesIO()
    plt.savefig(buf, format='png')
    buf.seek(0)
    chart_data = base64.b64encode(buf.getvalue()).decode('utf-8')
    buf.close()

    return render_template('charts.html', chart_data=chart_data)

if __name__ == "__main__":
    app.run(debug=True)
