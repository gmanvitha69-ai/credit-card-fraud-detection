from flask import (
    Flask,
    render_template,
    request,
    send_file,
    redirect,
    url_for,
    session
)
from reportlab.platypus import (
    SimpleDocTemplate,
    Paragraph,
    Spacer
)

from reportlab.lib.styles import (
    getSampleStyleSheet
)

import joblib
import numpy as np
import sqlite3
import matplotlib.pyplot as plt

app = Flask(__name__, template_folder='templates')

app.secret_key = "fraud_secret_key"

# Load AI model
model = joblib.load("models/fraud_model.pkl")

# Database connection
conn = sqlite3.connect(
    'fraud.db',
    check_same_thread=False
)

cursor = conn.cursor()

# Create predictions table
cursor.execute('''
CREATE TABLE IF NOT EXISTS predictions (

    id INTEGER PRIMARY KEY AUTOINCREMENT,

    time TEXT,

    amount TEXT,

    result TEXT
)
''')

conn.commit()

# =========================
# HOME PAGE
# =========================

@app.route('/')
def home():

    return render_template(
        'index.html'
    )

# =========================
# PREDICTION ROUTE
# =========================

@app.route('/predict', methods=['POST'])
def predict():

    try:

        # Get form values
        features = [
            float(x)
            for x in request.form.values()
        ]

        # Convert to numpy array
        final_features = np.array(
            features
        ).reshape(1, -1)

        # Predict
        prediction = model.predict(
            final_features
        )

        # Prediction result
        if prediction[0] == 1:

            result = (
                "Fraud Transaction Detected!"
            )

        else:

            result = (
                "Genuine Transaction"
            )

        # Save to database
        cursor.execute(
            """
            INSERT INTO predictions
            (time, amount, result)
            VALUES (?, ?, ?)
            """,
            (
                str(features[0]),
                str(features[4]),
                result
            )
        )

        conn.commit()

        return render_template(
            'index.html',
            prediction_text=result
        )

    except Exception as e:

        return render_template(
            'index.html',
            prediction_text=f"Error: {e}"
        )

# =========================
# HISTORY PAGE
# =========================

@app.route('/history')
def history():

    # Check login
    if 'user' not in session:

        return redirect(
            url_for('login')
        )

    cursor.execute(
        "SELECT * FROM predictions"
    )

    rows = cursor.fetchall()

    return render_template(
        'history.html',
        rows=rows
    )

# =========================
# DASHBOARD PAGE
# =========================

@app.route('/dashboard')
def dashboard():

    # Check login
    if 'user' not in session:

        return redirect(
            url_for('login')
        )

    cursor.execute(
        """
        SELECT result, COUNT(*)
        FROM predictions
        GROUP BY result
        """
    )

    data = cursor.fetchall()

    labels = [x[0] for x in data]
    values = [x[1] for x in data]

    plt.figure(figsize=(5,5))

    plt.pie(
        values,
        labels=labels,
        autopct='%1.1f%%'
    )

    plt.title(
        "Fraud vs Genuine Transactions"
    )

    plt.savefig(
        "app/static/chart.png"
    )

    return render_template(
        'dashboard.html'
    )

# =========================
# PDF REPORT DOWNLOAD
# =========================

@app.route('/download_report')
def download_report():

    try:

        import os

        # Current folder path
        base_dir = os.path.dirname(
            os.path.abspath(__file__)
        )

        # Full PDF path
        pdf_path = os.path.join(
            base_dir,
            "fraud_report.pdf"
        )

        # Create PDF
        doc = SimpleDocTemplate(
            pdf_path
        )

        styles = getSampleStyleSheet()

        elements = []

        # Title
        title = Paragraph(
            "Credit Card Fraud Detection Report",
            styles['Title']
        )

        elements.append(title)

        elements.append(
            Spacer(1, 20)
        )

        # Fetch records
        cursor.execute(
            "SELECT * FROM predictions"
        )

        rows = cursor.fetchall()

        # Add rows into PDF
        for row in rows:

            text = f"""
            ID: {row[0]}<br/>
            Time: {row[1]}<br/>
            Amount: {row[2]}<br/>
            Result: {row[3]}<br/><br/>
            """

            elements.append(
                Paragraph(
                    text,
                    styles['BodyText']
                )
            )

            elements.append(
                Spacer(1, 12)
            )

        # Build PDF
        doc.build(elements)

        # Send PDF
        return send_file(
            pdf_path,
            as_attachment=True
        )

    except Exception as e:

        return f"PDF Error: {e}"

@app.route('/login', methods=['GET', 'POST'])
def login():

    if request.method == 'POST':

        username = request.form['username']
        password = request.form['password']

        if (
            username == 'admin'
            and
            password == 'admin123'
        ):

            # Store session
            session['user'] = username

            return redirect(
                url_for('dashboard')
            )

        else:

            return render_template(
                'login.html',
                message='Invalid Username or Password'
            )

    return render_template('login.html')

@app.route('/logout')
def logout():

    session.pop('user', None)

    return redirect(
        url_for('login')
    )
# =========================
# RUN APP
# =========================

if __name__ == "__main__":

    app.run(debug=True)