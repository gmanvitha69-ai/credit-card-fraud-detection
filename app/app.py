from flask import (
    Flask, render_template, request,
    send_file, redirect, url_for, session, jsonify
)
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer
from reportlab.lib.styles import getSampleStyleSheet

import joblib
import numpy as np
import pandas as pd
import sqlite3
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import io, os

app = Flask(__name__, template_folder='templates')
app.secret_key = "fraud_secret_key"

# Load AI model
model = joblib.load("models/fraud_model.pkl")

# Database connection
conn = sqlite3.connect('fraud.db', check_same_thread=False)
cursor = conn.cursor()

# Create predictions table (added confidence column)
cursor.execute('''
CREATE TABLE IF NOT EXISTS predictions (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    time TEXT,
    amount TEXT,
    result TEXT,
    confidence TEXT
)
''')
conn.commit()

# =========================
# SAMPLE DATA (real dataset rows)
# =========================

FRAUD_SAMPLE = {
    "Time": "406", "V1": "-2.3122265", "V2": "1.9519984",
    "V3": "-1.6098154", "Amount": "239.93"
}

GENUINE_SAMPLE = {
    "Time": "0", "V1": "-1.3598071", "V2": "-0.0727812",
    "V3": "2.5363467", "Amount": "149.62"
}

# =========================
# HOME PAGE
# =========================

@app.route('/')
def home():
    return render_template('index.html')

# =========================
# SAMPLE DATA API
# =========================

@app.route('/sample/<sample_type>')
def get_sample(sample_type):
    if sample_type == 'fraud':
        return jsonify(FRAUD_SAMPLE)
    else:
        return jsonify(GENUINE_SAMPLE)

# =========================
# PREDICTION ROUTE (single)
# =========================

@app.route('/predict', methods=['POST'])
def predict():
    try:
        features = [float(x) for x in request.form.values()]
        final_features = np.array(features).reshape(1, -1)

        prediction = model.predict(final_features)[0]
        probability = model.predict_proba(final_features)[0]

        fraud_prob = round(float(probability[1]) * 100, 1)
        legit_prob = round(float(probability[0]) * 100, 1)

        if prediction == 1:
            result = "Fraud Transaction Detected!"
            risk = "High"
        else:
            result = "Genuine Transaction"
            risk = "Low" if fraud_prob < 35 else "Medium"

        confidence = f"{fraud_prob}%"

        cursor.execute(
            "INSERT INTO predictions (time, amount, result, confidence) VALUES (?, ?, ?, ?)",
            (str(features[0]), str(features[-1]), result, confidence)
        )
        conn.commit()

        return render_template(
            'index.html',
            prediction_text=result,
            fraud_prob=fraud_prob,
            legit_prob=legit_prob,
            risk=risk
        )

    except Exception as e:
        return render_template('index.html', prediction_text=f"Error: {e}")

# =========================
# CSV UPLOAD PREDICTION
# =========================

@app.route('/predict_csv', methods=['POST'])
def predict_csv():
    try:
        file = request.files['csv_file']
        df = pd.read_csv(file)

        predictions = model.predict(df)
        probabilities = model.predict_proba(df)[:, 1]

        df['Result'] = ['Fraud' if p == 1 else 'Genuine' for p in predictions]
        df['Fraud Probability'] = [f"{round(p*100,1)}%" for p in probabilities]
        df['Risk'] = [
            'High' if p > 0.6 else 'Medium' if p > 0.35 else 'Low'
            for p in probabilities
        ]

        # Save each to DB
        for i, row in df.iterrows():
            cursor.execute(
                "INSERT INTO predictions (time, amount, result, confidence) VALUES (?, ?, ?, ?)",
                (str(row.get('Time', 'N/A')), str(row.get('Amount', 'N/A')),
                 row['Result'], row['Fraud Probability'])
            )
        conn.commit()

        # Show only key columns in result
        display_cols = ['Time', 'Amount', 'Result', 'Fraud Probability', 'Risk']
        display_cols = [c for c in display_cols if c in df.columns]
        results = df[display_cols].to_dict('records')
        total = len(results)
        fraud_count = sum(1 for r in results if r['Result'] == 'Fraud')
        genuine_count = total - fraud_count

        return render_template(
            'csv_result.html',
            results=results,
            total=total,
            fraud_count=fraud_count,
            genuine_count=genuine_count
        )

    except Exception as e:
        return render_template('index.html', prediction_text=f"CSV Error: {e}")

# =========================
# HISTORY PAGE
# =========================

@app.route('/history')
def history():
    if 'user' not in session:
        return redirect(url_for('login'))

    cursor.execute("SELECT * FROM predictions ORDER BY id DESC")
    rows = cursor.fetchall()
    return render_template('history.html', rows=rows)

# =========================
# DASHBOARD PAGE
# =========================

@app.route('/dashboard')
def dashboard():
    if 'user' not in session:
        return redirect(url_for('login'))

    cursor.execute("SELECT result, COUNT(*) FROM predictions GROUP BY result")
    data = cursor.fetchall()

    labels = [x[0] for x in data] if data else ['No Data']
    values = [x[1] for x in data] if data else [1]

    colors = []
    for label in labels:
        if 'Fraud' in label:
            colors.append('#e74c3c')
        else:
            colors.append('#2ecc71')

    fig, ax = plt.subplots(figsize=(5, 5))
    ax.pie(values, labels=labels, autopct='%1.1f%%',
           colors=colors, startangle=140,
           wedgeprops={'edgecolor': 'white', 'linewidth': 2})
    ax.set_title("Fraud vs Genuine Transactions", fontsize=14, fontweight='bold')
    plt.tight_layout()
    plt.savefig("app/static/chart.png", transparent=True)
    plt.close()

    # Stats for dashboard
    cursor.execute("SELECT COUNT(*) FROM predictions")
    total = cursor.fetchone()[0]
    cursor.execute("SELECT COUNT(*) FROM predictions WHERE result LIKE '%Fraud%'")
    fraud_total = cursor.fetchone()[0]
    genuine_total = total - fraud_total

    return render_template(
        'dashboard.html',
        total=total,
        fraud_total=fraud_total,
        genuine_total=genuine_total
    )

# =========================
# PDF REPORT DOWNLOAD
# =========================

@app.route('/download_report')
def download_report():
    try:
        base_dir = os.path.dirname(os.path.abspath(__file__))
        pdf_path = os.path.join(base_dir, "fraud_report.pdf")
        doc = SimpleDocTemplate(pdf_path)
        styles = getSampleStyleSheet()
        elements = []

        elements.append(Paragraph("Credit Card Fraud Detection Report", styles['Title']))
        elements.append(Spacer(1, 20))

        cursor.execute("SELECT * FROM predictions")
        rows = cursor.fetchall()

        for row in rows:
            text = f"ID: {row[0]}<br/>Time: {row[1]}<br/>Amount: {row[2]}<br/>Result: {row[3]}<br/>Confidence: {row[4] if len(row) > 4 else 'N/A'}<br/><br/>"
            elements.append(Paragraph(text, styles['BodyText']))
            elements.append(Spacer(1, 12))

        doc.build(elements)
        return send_file(pdf_path, as_attachment=True)

    except Exception as e:
        return f"PDF Error: {e}"

# =========================
# LOGIN / LOGOUT
# =========================

@app.route('/login', methods=['GET', 'POST'])
def login():
    if request.method == 'POST':
        username = request.form['username']
        password = request.form['password']
        if username == 'admin' and password == 'admin123':
            session['user'] = username
            return redirect(url_for('dashboard'))
        else:
            return render_template('login.html', message='Invalid Username or Password')
    return render_template('login.html')

@app.route('/logout')
def logout():
    session.pop('user', None)
    return redirect(url_for('login'))

if __name__ == "__main__":
    app.run(debug=True)