from flask import Flask, render_template, request, redirect, url_for, session,flash
import pandas as pd
import matplotlib.pyplot as plt
import io
import base64
import joblib
import warnings
from sklearn.exceptions import InconsistentVersionWarning
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.ensemble import RandomForestClassifier

# Suppress warnings about version mismatches
warnings.filterwarnings("ignore", category=InconsistentVersionWarning)


app=Flask(__name__)

# Load model and vectorizer
vectorizer = joblib.load('vectorizer.pkl')
model = joblib.load('support_vector.pkl')

# Dataset path
dataset_path = 'hazard.csv'


@app.route("/" , methods=["GET", "POST"])

def index():
    
      if 'submit' in request.form:
        if request.method=="POST":
            uname=request.form.get("uname")
            pswd=request.form.get("pswd")
            
              
            if (uname == "admin" and pswd == "admin"):
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
    columns_to_display = ['url', 'type']  # Specify columns to display
    column_widths = {'url': '70%', 'type': '30%'}  # Adjust column widths as needed
    return render_template('result.html', columns=columns_to_display,  rows=df.to_dict(orient='records'))
    #return render_template('result.html', tables=[df.to_html(classes='table table-striped', escape=False)])

  
@app.route('/prediction', methods=['GET', 'POST'])
def predict():
    if request.method == 'POST':
        user_text = request.form['input_text']
        input_features = vectorizer.transform([user_text])
        prediction = model.predict(input_features)[0]
         # Correct label mapping
        label_map = {
            0: "phishing",
            1: "benign",
            2: "defacement",
            3: "malware"
        }
    
        
        result = label_map[prediction]
        flash(result)
        return render_template('prediction.html', input_text=user_text, result=result)
        
    return render_template('prediction.html', input_text=None, result=None)

@app.route('/charts')
def charts():
    df = pd.read_csv(dataset_path)
    label_counts = df['label'].value_counts()
    plt.bar(label_counts.index, label_counts.values)
    plt.xlabel('Labels')
    plt.ylabel('Counts')
    plt.title('Dataset Label Distribution')
    buf = io.BytesIO()
    plt.savefig(buf, format='png')
    buf.seek(0)
    chart_data = base64.b64encode(buf.getvalue()).decode('utf-8')
    buf.close()
    return render_template('charts.html', chart_data=chart_data)

if __name__=="__main__":
    app.secret_key="123"
    app.run(debug=True)