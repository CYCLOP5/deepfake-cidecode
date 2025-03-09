from flask import Flask, request, jsonify, send_file, redirect, logging
from flask_cors import CORS
import os
import sys

sys.path.append('/home/cyclops/Music/mri_gan_deepfake')
import runmodel

app = Flask(__name__)
CORS(app, supports_credentials=True)

UPLOAD_FOLDER = "uploads"
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

users = {}

@app.route("/register", methods=["POST"])
def register():
    data = request.json
    email = data.get("email")
    password = data.get("password")
    if email in users:
        return jsonify({"error": "User already exists"}), 400
    users[email] = password
    return jsonify({"message": "User registered successfully"}), 201

@app.route("/login", methods=["POST"])
def login():
    data = request.json
    email = data.get("email")
    password = data.get("password")
    if users.get(email) == password:
        return jsonify({"message": "Login successful"}), 200
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
            logging.exception("Error sending report file")
            return jsonify({"error": "Error sending report"}), 500
    else:
        logging.error("Report file not found")
        return jsonify({"error": "Report not found"}), 404



if __name__ == "__main__":
    app.run(debug=True, port=5000)

