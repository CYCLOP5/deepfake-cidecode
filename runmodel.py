import subprocess
import os
import shutil

def run_deepfake_detection_and_report(input_video):
    work_dir = "/home/cyclops/Music/mri_gan_deepfake"
    output_dir = os.path.join(work_dir, "output")

    try:
        subprocess.run(["python", "deep_fake_detect_app.py", "--input_file", input_video], 
                       check=True, cwd=work_dir)
        subprocess.run(["python", "report.py", "detection_output.json", "--dark-mode"], 
                       check=True, cwd=work_dir)
        
        if os.path.exists(output_dir):
            shutil.rmtree(output_dir)

        return True
    except subprocess.CalledProcessError as e:
        print(f"Error running deepfake detection pipeline: {e}")
        return False
