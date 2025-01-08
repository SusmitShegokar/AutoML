from flask import Flask, render_template, request, send_file, jsonify, session
import pandas as pd
import sqlite3
import os
import pickle
from ydata_profiling import ProfileReport
from sklearn.model_selection import train_test_split
import json
from werkzeug.utils import secure_filename

app = Flask(__name__)
app.secret_key = 'your-secret-key-here'  # Required for session management

# Directories
UPLOAD_FOLDER = 'uploads'
PROFILES_DIR = "profiles"
MODELS_DIR = "models"
ALLOWED_EXTENSIONS = {'csv'}

# Create necessary directories
for directory in [UPLOAD_FOLDER, PROFILES_DIR, MODELS_DIR]:
    os.makedirs(directory, exist_ok=True)

app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

def init_db(filename):
    conn = sqlite3.connect(f"{filename}.db")
    return conn

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/upload', methods=['POST'])
def upload_file():
    if 'file' not in request.files:
        return jsonify({'error': 'No file part'}), 400
    
    file = request.files['file']
    if file.filename == '':
        return jsonify({'error': 'No selected file'}), 400
    
    if file and allowed_file(file.filename):
        filename = secure_filename(file.filename)
        filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        file.save(filepath)
        
        # Read and store DataFrame
        df = pd.read_csv(filepath)
        
        # Store in database
        db_filename = filename.split('.')[0]
        conn = init_db(db_filename)
        df.to_sql("data", conn, if_exists="replace", index=False)
        
        # Store filename in session
        session['uploaded_filename'] = db_filename
        
        # Get column names for UI
        columns = df.columns.tolist()
        
        return jsonify({
            'success': True,
            'message': f'File uploaded successfully as {db_filename}.db',
            'columns': columns
        })
    
    return jsonify({'error': 'Invalid file type'}), 400

@app.route('/analyze', methods=['GET'])
def analyze_data():
    if 'uploaded_filename' not in session:
        return jsonify({'error': 'No file uploaded'}), 400
    
    filename = session['uploaded_filename']
    conn = init_db(filename)
    df = pd.read_sql_query("SELECT * FROM data", conn)
    
    # Generate profile report
    profile = ProfileReport(df, title="Pandas Profiling Report", explorative=True)
    profile_path = os.path.join(PROFILES_DIR, f"{filename}_profile.html")
    profile.to_file(profile_path)
    
    return jsonify({
        'success': True,
        'profile_path': profile_path
    })

@app.route('/train', methods=['POST'])
def train_models():
    if 'uploaded_filename' not in session:
        return jsonify({'error': 'No file uploaded'}), 400
    
    data = request.json
    target = data.get('target')
    features = data.get('features')
    framework = data.get('framework', 'scikit-learn')
    
    filename = session['uploaded_filename']
    conn = init_db(filename)
    df = pd.read_sql_query("SELECT * FROM data", conn)
    
    if not all([target, features]):
        return jsonify({'error': 'Missing target or features'}), 400
    
    X = df[features]
    y = df[target]
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    if framework == 'scikit-learn':
        from models import evaluate_models
        results = evaluate_models(X_train, X_test, y_train, y_test)
        
        # Save results
        results_path = os.path.join(MODELS_DIR, f"{filename}_results.pkl")
        with open(results_path, 'wb') as f:
            pickle.dump(results, f)
        
        return jsonify({
            'success': True,
            'results': results
        })
    
    elif framework == 'PyCaret':
        from models import pycaret_evaluation
        results, best_model = pycaret_evaluation(df, target)
        
        # Save results
        results_path = os.path.join(MODELS_DIR, f"{filename}_results.pkl")
        with open(results_path, 'wb') as f:
            pickle.dump({'results': results, 'model': best_model}, f)
        
        return jsonify({
            'success': True,
            'results': results
        })
    
    return jsonify({'error': 'Invalid framework specified'}), 400

@app.route('/download/<filename>')
def download_model(filename):
    try:
        return send_file(
            os.path.join(MODELS_DIR, filename),
            as_attachment=True,
            download_name=filename
        )
    except Exception as e:
        return jsonify({'error': str(e)}), 400

if __name__ == '__main__':
    app.run(debug=True)