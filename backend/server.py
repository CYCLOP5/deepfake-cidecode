from flask import Flask, request, jsonify,send_file
from flask_cors import CORS
import os
import sqlite3
import bcrypt
import os
import sys
sys.path.append('/home/cyclops/Music/mri_gan_deepfake')
import runmodel

app = Flask(__name__)
CORS(app, supports_credentials=True)
UPLOAD_FOLDER = "uploads"
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

DB_PATH = "users.db"

def init_db():
    with sqlite3.connect(DB_PATH) as conn:
        c = conn.cursor()
        c.execute("""
        CREATE TABLE IF NOT EXISTS users (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            email TEXT UNIQUE NOT NULL,
            password TEXT NOT NULL
        )
        """)
        conn.commit()

init_db()

@app.route("/register", methods=["POST"])
def register():
    data = request.json
    email = data.get("email")
    password = data.get("password")

    if not email or not password:
        return jsonify({"error": "Email and password required"}), 400

    hashed_password = bcrypt.hashpw(password.encode(), bcrypt.gensalt()).decode("utf-8")  # Store as string

    try:
        with sqlite3.connect(DB_PATH) as conn:
            c = conn.cursor()
            c.execute("INSERT INTO users (email, password) VALUES (?, ?)", (email, hashed_password))
            conn.commit()
        return jsonify({"message": "User registered successfully"}), 201
    except sqlite3.IntegrityError:
        return jsonify({"error": "User already exists"}), 400

@app.route("/login", methods=["POST"])
def login():
    data = request.json
    email = data.get("email")
    password = data.get("password")

    with sqlite3.connect(DB_PATH) as conn:
        c = conn.cursor()
        c.execute("SELECT password FROM users WHERE email = ?", (email,))
        user = c.fetchone()

        if user and bcrypt.checkpw(password.encode(), user[0].encode()):  # Convert stored password back to bytes
            return jsonify({"message": "Login successful", "redirect": "/upload"}), 200
        return jsonify({"error": "Invalid credentials"}), 401

@app.route("/upload", methods=["POST"])
def upload_video():
    if "file" not in request.files:
        return jsonify({"error": "No file uploaded"}), 400

    file = request.files["file"]
    if file.filename == "":
        return jsonify({"error": "No selected file"}), 400

    upload_folder_absolute = os.path.abspath(UPLOAD_FOLDER)
    file_path = os.path.join(upload_folder_absolute, file.filename)
    file.save(file_path)

    runmodel.run_deepfake_detection_and_report(file_path)
    return jsonify({"message": "File uploaded successfully", "filename": file.filename}), 200

@app.route("/report")
def report():
    report_path = "/home/cyclops/Music/mri_gan_deepfake/report.html"
    if os.path.exists(report_path):
        try:
            return send_file(report_path)
        except Exception as e:
            return jsonify({"error": "Error sending report"}), 500
    else:
        return jsonify({"error": "Report not found"}), 404



if __name__ == "__main__":
    app.run(debug=True, port=5000)

