from flask import Flask, request, jsonify, render_template, redirect, url_for
import os
import subprocess

app = Flask(__name__)

UPLOAD_FOLDER = 'uploads'
ALLOWED_EXTENSIONS = {'mp4', 'avi', 'mov', 'mp3', 'wav'}
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

# Ensure upload folder exists
if not os.path.exists(UPLOAD_FOLDER):
    os.makedirs(UPLOAD_FOLDER)

def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

@app.route("/")
def home():
    return render_template("index.html")

@app.route("/upload", methods=["GET", "POST"])
def upload_file():
    if request.method == "POST":
        if 'file' not in request.files:
            return jsonify({"error": "No file part"}), 400
        file = request.files['file']
        if file.filename == '':
            return jsonify({"error": "No selected file"}), 400

        if file and allowed_file(file.filename):
            filename = file.filename
            filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
            file.save(filepath)

            if filename.rsplit('.', 1)[1].lower() in {'mp3', 'wav'}:
                return redirect(url_for("process_audio", filename=filename))
            else:
                return redirect(url_for("process_video", filename=filename))
        else:
            return jsonify({"error": "File type not allowed"}), 400

    return render_template("upload.html")

@app.route("/upload/audio/<filename>")
def process_audio(filename):
    filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)

    try:
        result = subprocess.run(
            ["python", "detect_deepfake_audio.py", filepath],
            capture_output=True, text=True
        )
        report = result.stdout
    except Exception as e:
        return jsonify({"error": str(e)}), 500

    return jsonify({"report": report})

@app.route("/upload/video/<filename>")
def process_video(filename):
    filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)

    try:
        result = subprocess.run(
            ["python", "detect_deepfake_video.py", filepath],
            capture_output=True, text=True
        )
        report = result.stdout
    except Exception as e:
        return jsonify({"error": str(e)}), 500

    return jsonify({"report": report})

if __name__ == "__main__":
    app.run(debug=True)
