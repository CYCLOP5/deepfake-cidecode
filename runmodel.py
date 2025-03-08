import subprocess

def run_deepfake_detection_and_report(input_video):
    try:
        subprocess.run(["python", "deep_fake_detect_app.py", "--input_file", input_video], check=True)
        subprocess.run(["python", "report.py", "detection_output.json","--dark-mode"], check=True)
        return True
    except subprocess.CalledProcessError as e:
        print(f"Error running deepfake detection pipeline: {e}")
        return False
