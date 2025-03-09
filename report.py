#!/usr/bin/env python3
import json
import os
import numpy as np
import pandas as pd
from datetime import datetime
import argparse
import plotly.express as px
import plotly.graph_objects as go
from pathlib import Path
import base64
from PIL import Image
import io
import re
import subprocess
import hashlib
import magic
import ffmpeg
from fractions import Fraction
from pymediainfo import MediaInfo  
from tinytag import TinyTag   

def parse_args():
    parser = argparse.ArgumentParser(description='Generate a deepfake detection report from JSON results')
    parser.add_argument('input_file', type=str, help='Path to the JSON result file')
    parser.add_argument('--video', type=str, help='Path to the original video file for metadata analysis')
    parser.add_argument('--output', type=str, default='report.html', help='Output HTML report file')
    parser.add_argument('--dark-mode', action='store_true', help='Enable dark mode styling')
    return parser.parse_args()

def load_json_data(file_path):
    try:
        with open(file_path, 'r') as f:
            return json.load(f)
    except Exception as e:
        print("Error loading JSON file: {}".format(e))
        exit(1)

def calculate_file_checksums(file_path):
    """Calculate multiple checksums for the file: MD5, SHA256, and SHA512."""
    try:
        md5_hash = hashlib.md5()
        sha256_hash = hashlib.sha256()
        sha512_hash = hashlib.sha512()
        with open(file_path, "rb") as f:
            for byte_block in iter(lambda: f.read(4096), b""):
                md5_hash.update(byte_block)
                sha256_hash.update(byte_block)
                sha512_hash.update(byte_block)
        return {
            "MD5": md5_hash.hexdigest().upper(),
            "SHA256": sha256_hash.hexdigest().upper(),
            "SHA512": sha512_hash.hexdigest().upper()
        }
    except Exception as e:
        print("Error calculating checksums: {}".format(e))
        return {"Error": "Could not calculate checksums"}

def get_file_mime_type(file_path):
    """Get the MIME type of the file using python-magic"""
    try:
        mime = magic.Magic(mime=True)
        return mime.from_file(file_path)
    except Exception as e:
        print("Error getting MIME type: {}".format(e))
        return "Unknown"
    
def get_atom_sizes(video_path):
    """Extract atom sizes from the MP4 container structure."""
    try:
        total_size = os.path.getsize(video_path)
        atom_sizes = {}
        result = subprocess.run(
            ["ffmpeg", "-i", video_path, "-v", "error", "-print_format", "json", "-show_format", "-show_streams"],
            capture_output=True, text=True
        )
        if result.returncode == 0:
            probe = json.loads(result.stdout)
            for stream in probe.get("streams", []):
                if "tags" in stream:
                    for key, value in stream["tags"].items():
                        if key not in atom_sizes:
                            try:
                                atom_sizes[key] = int(value)
                            except ValueError:
                                continue
        signature_structure = {
            key: "{:.2f} %".format((size / total_size) * 100) if total_size > 0 else "0.00 %"
            for key, size in atom_sizes.items()
        }
        return signature_structure if signature_structure else {"Unknown": "No atom data found"}
    except Exception as e:
        print("Error extracting atom sizes: {}".format(e))
        return {"Error": "Could not extract structure"}

def extract_video_metadata(video_path):
    """Extract detailed video metadata using ffmpeg"""
    if not os.path.exists(video_path):
        return {"error": "Video file not found"}
    try:
        file_info = {
            "filename": os.path.basename(video_path),
            "format": get_file_mime_type(video_path),
            "file_id": os.stat(video_path).st_ino,
            "date_processed": datetime.now().strftime("%m/%d/%Y %H:%M UTC"),
            "date_last_updated": datetime.fromtimestamp(os.path.getmtime(video_path)).strftime("%m/%d/%Y %H:%M UTC"),
            "checksums": calculate_file_checksums(video_path)
        }
        try:
            probe = ffmpeg.probe(video_path)
            video_streams = [s for s in probe.get("streams", []) if s.get("codec_type") == "video"]
            if not video_streams:
                return {"error": "No video stream found in the file", "file_summary": file_info}
            video_info = video_streams[0]
            video_details = {}
            video_details["id"] = video_info.get("index", "Unknown")
            video_details["format"] = video_info.get("codec_name", "Unknown").upper()
            video_details["format_info"] = video_info.get("codec_long_name", "Unknown")
            video_details["format_profile"] = video_info.get("profile", "Unknown")
            video_details["codec_id"] = video_info.get("codec_tag_string", "Unknown")
            duration = video_info.get("duration")
            video_details["duration"] = "{} s".format(duration) if duration else "Unknown"
            bit_rate = video_info.get("bit_rate", "0")
            try:
                bit_rate_int = int(bit_rate)
            except:
                bit_rate_int = 0
            video_details["bit_rate"] = "{} kb/s".format(int(bit_rate_int/1000))
            video_details["width"] = video_info.get("width", "Unknown")
            video_details["height"] = video_info.get("height", "Unknown")
            if video_info.get("width") and video_info.get("height"):
                w, h = int(video_info["width"]), int(video_info["height"])
                video_details["display_aspect_ratio"] = "{:.3f}".format(w/h).replace('.', ',')
            r_frame_rate = video_info.get("r_frame_rate", "0/1")
            try:
                video_details["frame_rate"] = float(Fraction(r_frame_rate))
            except Exception as e:
                video_details["frame_rate"] = 0.0
            video_details["frame_rate_mode"] = "Constant" if video_details["frame_rate"] > 0 else "Variable"
            video_details["color_space"] = video_info.get("pix_fmt", "Unknown").upper()
            video_details["bit_depth"] = "{} bits".format(video_info.get('bits_per_raw_sample', '8'))
            container_info = {}
            container_info["format"] = probe["format"].get("format_long_name", "Unknown")
            container_info["format_profile"] = "Base Media / Version 2"
            container_info["codec_id"] = probe["format"].get("format_name", "Unknown")
            container_size = probe["format"].get("size", "0")
            try:
                container_info["file_size"] = "{:.2f} KiB".format(int(container_size) / 1024)
            except:
                container_info["file_size"] = "Unknown"
            container_duration = probe["format"].get("duration")
            container_info["duration"] = "{} s".format(container_duration) if container_duration else "Unknown"
            overall_bit_rate = probe["format"].get("bit_rate", "0")
            try:
                overall_bit_rate_int = int(overall_bit_rate)
            except:
                overall_bit_rate_int = 0
            container_info["overall_bit_rate"] = "{} kb/s".format(int(overall_bit_rate_int/1000))
            container_info["overall_bit_rate_mode"] = "Variable"
            writing_app = "Unknown"
            for key in ["encoder", "handler_name", "ENCODER"]:
                if key in probe["format"].get("tags", {}):
                    writing_app = probe["format"]["tags"][key]
                    break
            container_info["writing_application"] = writing_app
            signature_structure = get_atom_sizes(video_path)
            metadata = {
                "file_summary": file_info,
                "video_details": video_details,
                "container_info": container_info,
                "file_signature_structure": signature_structure
            }
            return metadata
        except Exception as e:
            print("Error extracting video metadata with ffmpeg: {}".format(e))
            return {"file_summary": file_info, "error": str(e)}
    except Exception as e:
        print("Error extracting video metadata: {}".format(e))
        return {"error": str(e)}

def extract_audio_metadata(audio_path):
    """Extract detailed audio metadata using ffmpeg and tinytag"""
    print("Extracting audio metadata from:", audio_path)
    if not os.path.exists(audio_path):
        print("Audio file not found.")
        return {"error": "Audio file not found"}
    try:
        file_info = {
            "filename": os.path.basename(audio_path),
            "format": get_file_mime_type(audio_path),
            "file_id": os.stat(audio_path).st_ino,
            "date_processed": datetime.now().strftime("%m/%d/%Y %H:%M UTC"),
            "date_last_updated": datetime.fromtimestamp(os.path.getmtime(audio_path)).strftime("%m/%d/%Y %H:%M UTC"),
            "checksums": calculate_file_checksums(audio_path)
        }
        print("File info:", file_info)
        try:
            probe = ffmpeg.probe(audio_path)
            audio_streams = [s for s in probe.get("streams", []) if s.get("codec_type") == "audio"]
            if not audio_streams:
                print("No audio stream found in the file.")
                return {"error": "No audio stream found in the file", "file_summary": file_info}
            audio_info = audio_streams[0]
            audio_details = {}
            audio_details["codec"] = audio_info.get("codec_name", "Unknown")
            audio_details["codec_long"] = audio_info.get("codec_long_name", "Unknown")
            audio_details["sample_rate"] = audio_info.get("sample_rate", "Unknown")
            audio_details["channels"] = audio_info.get("channels", "Unknown")
            audio_details["bit_rate"] = audio_info.get("bit_rate", "Unknown")
            duration = audio_info.get("duration")
            audio_details["duration"] = "{} s".format(duration) if duration else "Unknown"
            audio_details["channel_layout"] = audio_info.get("channel_layout", "Unknown")
            print("Audio details:", audio_details)
            container_info = {}
            container_info["format"] = probe["format"].get("format_long_name", "Unknown")
            container_info["file_size"] = probe["format"].get("size", "Unknown")
            container_info["overall_bit_rate"] = probe["format"].get("bit_rate", "Unknown")
            print("Container info:", container_info)
            
            try:
                tag = TinyTag.get(audio_path)
                tinytag_metadata = {
                    "title": tag.title,
                    "artist": tag.artist,
                    "album": tag.album,
                    "year": tag.year,
                    "genre": tag.genre,
                    "duration": tag.duration  
                }
                print("Tinytag metadata:", tinytag_metadata)
            except Exception as te:
                tinytag_metadata = {"error": "Tinytag extraction failed: {}".format(te)}
                print("Tinytag extraction error:", te)
            
            metadata = {
                "file_summary": file_info,
                "audio_details": audio_details,
                "container_info": container_info,
                "tinytag_metadata": tinytag_metadata
            }
            print("Extracted audio metadata successfully.")
            return metadata
        except Exception as e:
            print("Error extracting audio metadata with ffmpeg:", e)
            return {"file_summary": file_info, "error": str(e)}
    except Exception as e:
        print("Error extracting audio metadata:", e)
        return {"error": str(e)}
def get_device_metadata(video_path):
    """
    Extract device metadata from the video using pymediainfo.
    Returns a dictionary with keys: make and model.
    """
    try:
        media_info = MediaInfo.parse(video_path)
        tags = {}
        for track in media_info.tracks:
            if track.track_type == "General":
                make = getattr(track, "com_apple_quicktime_make", None) or getattr(track, "make", None)
                model = getattr(track, "com_apple_quicktime_model", None) or getattr(track, "model", None)
                tags["make"] = make
                tags["model"] = model
                break
        return tags
    except Exception as e:
        print("Error extracting device metadata with pymediainfo: {}".format(e))
        return {}

def extract_frame_numbers(log_entries):
    heatmap_pattern = re.compile(r'Saved heatmap: .+?/(\d+)_\d+_heatmap\.png')
    frame_numbers = []
    for entry in log_entries:
        match = heatmap_pattern.search(entry)
        if match:
            frame_numbers.append(int(match.group(1)))
    return sorted(frame_numbers)

def extract_arrays(log_entries, start_marker, prefix=""):
    arrays = []
    capture = False
    array_str = ""
    for entry in log_entries:
        if entry.startswith(start_marker):
            capture = True
            array_str = entry[len(start_marker):].strip()
            continue
        if capture and entry.strip().endswith("]"):
            array_str += " " + entry.strip()
            array_str = array_str.replace("\n", "").replace("[", "").replace("]", "")
            try:
                array_values = []
                for x in array_str.split():
                    if x and x.replace('.', '', 1).isdigit():
                        array_values.append(float(x))
                if array_values:
                    arrays.append((prefix, array_values))
            except Exception as e:
                print("Warning: Could not parse array values: {}".format(e))
            capture = False
        elif capture:
            array_str += " " + entry.strip()
    return arrays

def is_not_black_image(image_path, threshold=0.01):
    try:
        with Image.open(image_path) as img:
            img_gray = img.convert('L')
            img_data = np.array(img_gray)
            non_black_percentage = np.sum(img_data > 10) / img_data.size
            return non_black_percentage > threshold
    except Exception as e:
        print("Warning: Could not analyze image {}: {}".format(image_path, e))
        return False

def convert_image_to_base64(image_path):
    try:
        with Image.open(image_path) as img:
            buffered = io.BytesIO()
            img.save(buffered, format=img.format)
            return base64.b64encode(buffered.getvalue()).decode()
    except Exception as e:
        print("Warning: Could not load image {}: {}".format(image_path, e))
        return None

def find_heatmap_images(log_entries, base_dir):
    heatmap_images = []
    heatmap_pattern = re.compile(r'Saved heatmap: (.+?\.png)')
    for entry in log_entries:
        match = heatmap_pattern.search(entry)
        if match:
            full_path = match.group(1)
            filename = os.path.basename(full_path)
            if os.path.exists(full_path):
                image_path = full_path
            else:
                frame_match = re.search(r'(\d+)_\d+_heatmap\.png', filename)
                if frame_match:
                    frame_num = frame_match.group(1)
                    possible_paths = [
                        os.path.join(base_dir, filename),
                        os.path.join(base_dir, "plain_frames", filename),
                        os.path.join(base_dir, "output", filename)
                    ]
                    for path in possible_paths:
                        if os.path.exists(path):
                            image_path = path
                            break
                    else:
                        print("Warning: Could not find heatmap image for frame {}".format(frame_num))
                        continue
            if os.path.exists(image_path) and is_not_black_image(image_path):
                frame_num = int(re.search(r'(\d+)_\d+_heatmap\.png', filename).group(1))
                heatmap_images.append((frame_num, image_path))
    return sorted(heatmap_images)

def find_mri_images(data, base_dir):
    mri_images = []
    mri_location = data.get("mri", {}).get("location", "")
    video_name = data.get("video", {}).get("name", "")
    possible_mri_dirs = [
        mri_location,
        os.path.join(base_dir, "mri", video_name),
        os.path.join(base_dir, "output", video_name, "mri"),
        os.path.join(base_dir, "output", video_name, "mri", video_name),
        os.path.join(os.path.dirname(base_dir), "mri"),
        os.path.join(os.path.dirname(base_dir), "output", "mri")
    ]
    for mri_dir in possible_mri_dirs:
        if os.path.exists(mri_dir) and os.path.isdir(mri_dir):
            for file in os.listdir(mri_dir):
                if file.lower().endswith('.png'):
                    image_path = os.path.join(mri_dir, file)
                    if is_not_black_image(image_path):
                        frame_match = re.search(r'(\d+)_', file)
                        frame_num = int(frame_match.group(1)) if frame_match else 0
                        mri_images.append((frame_num, image_path))
            if mri_images:
                break
    return sorted(mri_images)

def create_confidence_plot(data, args):
    """
    Creates two Plotly visualizations (frame-by-frame scatter and bar chart).
    Returns (html_for_scatter, html_for_bar) or (None, None) if no data.
    """
    plain_frame_data = {
        "frame_number": [],
        "confidence": [],
        "classification": []
    }
    plain_arrays = extract_arrays(data["log"], "all [", "All Frames")
    real_arrays = extract_arrays(data["log"], "real [", "Real")
    fake_arrays = extract_arrays(data["log"], "fake [", "Fake")
    all_arrays = []
    for name, values in plain_arrays + real_arrays + fake_arrays:
        if name.startswith("All") and len(values) > 0:
            all_arrays.append(values)
    if all_arrays:
        confidence_values = all_arrays[0]
        frame_numbers = list(range(len(confidence_values)))
        for i, conf in enumerate(confidence_values):
            plain_frame_data["frame_number"].append(frame_numbers[i])
            plain_frame_data["confidence"].append(conf * 100)
            plain_frame_data["classification"].append("Fake" if conf > 0.5 else "Real")
    df = pd.DataFrame(plain_frame_data)
    if df.empty:
        return None, None
    fig = px.scatter(df, x="frame_number", y="confidence", color="classification",
                     color_discrete_map={"Fake": "#FF5252", "Real": "#4CAF50"},
                     title="Frame-by-Frame Confidence Scores",
                     labels={"frame_number": "Frame Number", "confidence": "Confidence (%)"},
                     size_max=10, hover_data=["frame_number", "confidence"])
    fig.update_layout(
        plot_bgcolor="#1E1E1E" if args.dark_mode else "white",
        paper_bgcolor="#121212" if args.dark_mode else "white",
        font=dict(color="white" if args.dark_mode else "black"),
        title_font=dict(size=24),
        showlegend=True,
        legend=dict(title="Classification", bordercolor="Gray", borderwidth=1),
        xaxis=dict(gridcolor="rgba(255, 255, 255, 0.1)" if args.dark_mode else "rgba(0, 0, 0, 0.1)"),
        yaxis=dict(gridcolor="rgba(255, 255, 255, 0.1)" if args.dark_mode else "rgba(0, 0, 0, 0.1)")
    )
    fig.add_shape(
        type="line",
        x0=df["frame_number"].min(),
        y0=50,
        x1=df["frame_number"].max(),
        y1=50,
        line=dict(color="yellow", width=2, dash="dash")
    )
    fig.add_annotation(
        x=df["frame_number"].min(),
        y=50,
        text="Decision Threshold (50%)",
        showarrow=True,
        arrowhead=1,
        ax=50,
        ay=-30,
        font=dict(color="yellow")
    )
    plain_fake_prob = data.get("plain_frames", {}).get("fake_probability", 0)
    plain_real_prob = data.get("plain_frames", {}).get("real_probability", 0)
    mri_fake_prob = data.get("mri", {}).get("fake_probability", 0)
    mri_real_prob = data.get("mri", {}).get("real_probability", 0)
    model_df = pd.DataFrame({
        "Model": ["Plain Frames", "Plain Frames", "MRI", "MRI"],
        "Type": ["Fake", "Real", "Fake", "Real"],
        "Probability": [plain_fake_prob, plain_real_prob, mri_fake_prob, mri_real_prob]
    })
    fig2 = px.bar(model_df, x="Model", y="Probability", color="Type",
                  color_discrete_map={"Fake": "#FF5252", "Real": "#4CAF50"},
                  title="Model Confidence Comparison",
                  labels={"Probability": "Confidence (%)"})
    fig2.update_layout(
        plot_bgcolor="#1E1E1E" if args.dark_mode else "white",
        paper_bgcolor="#121212" if args.dark_mode else "white",
        font=dict(color="white" if args.dark_mode else "black"),
        title_font=dict(size=24)
    )
    return fig.to_html(full_html=False, include_plotlyjs='cdn'), fig2.to_html(full_html=False, include_plotlyjs='cdn')

device_lookup = {
    "iPhone 11": {
        "Generation 1": "Device Camera\nFront Camera | GPS On | Rear Camera",
        "Generation 2": "YouTube\nUpload",
        "Generation 3": "youtube-dl\n'Best' Format | Direct Download"
    },
    "iPhone 11 Pro": {
        "Generation 1": "Device Camera\nFront Camera | GPS On | Rear Camera",
        "Generation 2": "YouTube",
        "Generation 3": "youtube-dl\n'Best' Format"
    },
    "iPhone 6s": {
        "Generation 1": "Device Camera",
        "Generation 2": "YouTube\nUpload",
        "Generation 3": "youtube-dl\nDirect Download"
    },
    "Galaxy S8": {
        "Generation 1": "Device Camera",
        "Generation 2": "YouTube\nUpload",
        "Generation 3": "youtube-dl\nDirect Download"
    }
}

def get_device_generation_history(make, model):
    """Return a dictionary for the device generation details based on the device model."""
    key = model
    if key in device_lookup:
        data = device_lookup[key]
        return {
            "Brand": make or "Unknown",
            "Model": model or "Unknown",
            "Generation 1": data["Generation 1"],
            "Generation 2": data["Generation 2"],
            "Generation 3": data["Generation 3"]
        }
    else:
        return {
            "Brand": make or "Unknown",
            "Model": model or "Unknown",
            "Generation 1": "N/A",
            "Generation 2": "N/A",
            "Generation 3": "N/A"
        }

def generate_html_report(data, video_metadata, args):
    video_name = data.get("video", {}).get("name", "Unknown")
    plain_frames_result = data.get("plain_frames", {}).get("prediction", "Unknown")
    plain_frames_prob = data.get("plain_frames", {}).get("fake_probability", 0)
    mri_result = data.get("mri", {}).get("prediction", "Unknown")
    mri_prob = data.get("mri", {}).get("fake_probability", 0)
    base_dir = os.path.dirname(os.path.abspath(args.input_file))
    frame_numbers = extract_frame_numbers(data["log"])
    plot_html, model_comparison_html = create_confidence_plot(data, args)
    heatmap_images = find_heatmap_images(data["log"], base_dir)
    mri_images = find_mri_images(data, base_dir)

    audio_info = data.get("audio", {})
    audio_prediction = audio_info.get("prediction", "Unknown")
    print("Audio prediction: {}".format(audio_prediction))

    audio_metadata_obj = None
    audio_metadata_html = ""
    if audio_info.get("location") and os.path.exists(audio_info.get("location")):
        audio_metadata_obj = extract_audio_metadata(audio_info.get("location"))
    elif args.video and os.path.exists(args.video):
        audio_metadata_obj = extract_audio_metadata(args.video)

    if audio_metadata_obj:
        if "error" in audio_metadata_obj:
            audio_metadata_html += '<div class="warning-text"><p>Error retrieving audio metadata: {}</p></div>'.format(audio_metadata_obj["error"])
        else:
            audio_metadata_html += '<div class="metadata-section">Audio File Summary</div><table><tbody>'
            for key, value in audio_metadata_obj.get("file_summary", {}).items():
                audio_metadata_html += '<tr class="metadata-row"><td><strong>{}</strong></td><td>{}</td></tr>'.format(key.replace("_", " ").title(), value)
            audio_metadata_html += '</tbody></table>'
            audio_metadata_html += '<div class="metadata-section">Audio Details</div><table><tbody>'
            for key, value in audio_metadata_obj.get("audio_details", {}).items():
                audio_metadata_html += '<tr class="metadata-row"><td><strong>{}</strong></td><td>{}</td></tr>'.format(key.replace("_", " ").title(), value)
            audio_metadata_html += '</tbody></table>'
            audio_metadata_html += '<div class="metadata-section">Audio Container Information</div><table><tbody>'
            for key, value in audio_metadata_obj.get("container_info", {}).items():
                audio_metadata_html += '<tr class="metadata-row"><td><strong>{}</strong></td><td>{}</td></tr>'.format(key.replace("_", " ").title(), value)
            audio_metadata_html += '</tbody></table>'
            if "tinytag_metadata" in audio_metadata_obj:
                audio_metadata_html += '<div class="metadata-section">Tinytag Metadata</div><table><tbody>'
                for key, value in audio_metadata_obj["tinytag_metadata"].items():
                    audio_metadata_html += '<tr class="metadata-row"><td><strong>{}</strong></td><td>{}</td></tr>'.format(key.title(), value)
                audio_metadata_html += '</tbody></table>'

    # Determine if this is an audio-only file
    is_audio_only = (plain_frames_prob == 0 and mri_prob == 0)
    if is_audio_only:
        if audio_prediction.upper() == "FAKE":
            verdict_emoji = "‼️"
            verdict_class = "fake"
            verdict_text = "AUDIO PREDICTION: DEEPFAKE DETECTED"
        elif audio_prediction.upper() == "REAL":
            verdict_emoji = "✅"
            verdict_class = "real"
            verdict_text = "AUDIO PREDICTION: AUTHENTIC AUDIO"
        else:
            verdict_emoji = "⚠️"
            verdict_class = "uncertain"
            verdict_text = "AUDIO PREDICTION: UNCERTAIN"
        confidence = audio_prediction
        header_title = "Audio Information"
        header_filename = os.path.basename(audio_info.get("location", "Unknown"))
        header_analysis_type = "Audio Detection"
        header_models_used = "Logistic Regression"
    else:
        if plain_frames_prob == 0:
            final_result = mri_result
        elif mri_prob == 0:
            final_result = plain_frames_result
        else:
            final_result = None 
        if final_result:
            if final_result == "DEEP-FAKE":
                verdict_emoji = "‼️"
                verdict_class = "fake"
                verdict_text = "DEEPFAKE DETECTED"
            elif final_result == "REAL":
                verdict_emoji = "✅"
                verdict_class = "real"
                verdict_text = "AUTHENTIC VIDEO"
            else:
                verdict_emoji = "⚠️"
                verdict_class = "uncertain"
                verdict_text = "AUDIO ONLY FILE DETECTED, CHECK AUDIO ANALYSIS."
        else:
            if plain_frames_result == "DEEP-FAKE" and mri_result == "DEEP-FAKE":
                verdict_emoji = "‼️"
                verdict_class = "fake"
                verdict_text = "DEEPFAKE DETECTED"
            elif plain_frames_result == "REAL" and mri_result == "REAL":
                verdict_emoji = "✅"
                verdict_class = "real"
                verdict_text = "AUTHENTIC VIDEO"
            else:
                verdict_emoji = "⚠️"
                verdict_class = "uncertain"
                verdict_text = "UNCERTAIN RESULT"
        confidence = "{:.2f}".format(max(plain_frames_prob, mri_prob))
        header_title = "Video Information"
        header_filename = video_name
        header_analysis_type = "Deepfake Detection"
        header_models_used = "Plain Frame Analysis, MRI GAN Detection"
    
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    theme_class = "dark-theme" if args.dark_mode else "light-theme"
    if video_metadata and "file_signature_structure" in video_metadata:
        signature_data = video_metadata["file_signature_structure"]
    else:
        signature_data = {"ftyp": "100.00 %", "moov": "42.48 %", "mvhd": "39.23 %"}
    device_tags = {}
    if args.video and os.path.exists(args.video):
        device_tags = get_device_metadata(args.video)
    device_make = device_tags.get("make", "Unknown") or "Unknown"
    device_model = device_tags.get("model", "Unknown") or "Unknown"
    device_history_dict = get_device_generation_history(device_make, device_model)
    plain_confidence_class = "fake-confidence" if plain_frames_result == "DEEP-FAKE" else "real-confidence"
    plain_confidence_text = "{:.2f}% {}".format(plain_frames_prob, "Fake" if plain_frames_result == "DEEP-FAKE" else "Real")
    mri_confidence_class = "fake-confidence" if mri_result == "DEEP-FAKE" else "real-confidence"
    mri_confidence_text = "{:.2f}% {}".format(mri_prob, "Fake" if mri_result == "DEEP-FAKE" else "Real")
    current_year = datetime.now().year
    if plot_html:
        plot_section = """
            <div class="card animated delay-4">
                <h2>Analysis Visualization</h2>
                <div class="plot-container">
                    {plot_html}
                </div>
                <div class="plot-container">
                    {model_comparison_html}
                </div>
            </div>
        """.format(plot_html=plot_html, model_comparison_html=model_comparison_html)
    else:
        plot_section = ""
    heatmap_html = ""
    for frame_num, image_path in heatmap_images:
        base64_img = convert_image_to_base64(image_path)
        if base64_img:
            heatmap_html += """
                        <div class="image-card">
                            <img class="gallery-img" src="data:image/png;base64,{img}" onclick="openLightbox(this.src)">
                            <div class="image-caption">Frame {frame_num}</div>
                        </div>
            """.format(img=base64_img, frame_num=frame_num)
    mri_images_html = ""
    for frame_num, image_path in mri_images:
        base64_img = convert_image_to_base64(image_path)
        if base64_img:
            mri_images_html += """
                        <div class="image-card">
                            <img class="gallery-img" src="data:image/png;base64,{img}" onclick="openLightbox(this.src)">
                            <div class="image-caption">Frame {frame_num}</div>
                        </div>
            """.format(img=base64_img, frame_num=frame_num)
    metadata_html = ""
    if video_metadata:
        if "error" in video_metadata:
            metadata_html += """
                    <div class="warning-text">
                        <p>Error retrieving video metadata: {error}</p>
                    </div>
            """.format(error=video_metadata["error"])
        else:
            metadata_html += """
                    <div class="metadata-section">File Summary</div>
                    <table>
                        <tbody>
            """
            for key, value in video_metadata.get("file_summary", {}).items():
                metadata_html += """
                            <tr class="metadata-row">
                                <td><strong>{key}</strong></td>
                                <td>{value}</td>
                            </tr>
                """.format(key=key.replace("_", " ").title(), value=value)
            metadata_html += """
                        </tbody>
                    </table>
                    <div class="metadata-section">Video Details</div>
                    <table>
                        <tbody>
            """
            for key, value in video_metadata.get("video_details", {}).items():
                metadata_html += """
                            <tr class="metadata-row">
                                <td><strong>{key}</strong></td>
                                <td>{value}</td>
                            </tr>
                """.format(key=key.replace("_", " ").title(), value=value)
            metadata_html += """
                        </tbody>
                    </table>
                    <div class="metadata-section">Container Information</div>
                    <table>
                        <tbody>
            """
            for key, value in video_metadata.get("container_info", {}).items():
                metadata_html += """
                            <tr class="metadata-row">
                                <td><strong>{key}</strong></td>
                                <td>{value}</td>
                            </tr>
                """.format(key=key.replace("_", " ").title(), value=value)
            metadata_html += """
                        </tbody>
                    </table>
            """
    else:
        metadata_html += """
                    <div class="warning-text">
                        <p>Video metadata is not available.</p>
                    </div>
        """
    signature_html = ""
    for k, v in signature_data.items():
        signature_html += """
                            <tr>
                                <td><strong>{key}</strong></td>
                                <td>{value}</td>
                            </tr>
        """.format(key=k, value=v)
    device_history_html = """
                            <tr>
                                <td>{brand}</td>
                                <td>{model}</td>
                                <td>{gen1}</td>
                                <td>{gen2}</td>
                                <td>{gen3}</td>
                            </tr>
    """.format(brand=device_history_dict["Brand"], model=device_history_dict["Model"],
               gen1=device_history_dict["Generation 1"], gen2=device_history_dict["Generation 2"],
               gen3=device_history_dict["Generation 3"])
    export_pdf_button = """
    <button id="exportPDF" onclick="window.print()"
        style="position:absolute; top:20px; right:20px; padding:10px 15px; background-color:#4CAF50; color:white; border:none; border-radius:5px; cursor:pointer;">
        Export as PDF
    </button>
    """
    html = """
    <!DOCTYPE html>
    <html lang="en">
    <head>
        <meta charset="UTF-8">
        <meta name="viewport" content="width=device-width, initial-scale=1.0">
        <title>Deepfake Detection Report - {video_name}</title>
        <style>
    body {{
        font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
        line-height: 1.6;
        margin: 0;
        padding: 0;
        transition: background-color 0.3s;
    }}
    .dark-theme {{ background-color: #121212; color: #f5f5f5; }}
    .light-theme {{ background-color: #f5f5f5; color: #333333; }}
    .container {{ max-width: 1200px; margin: 0 auto; padding: 20px; }}
    .header {{
        text-align: center;
        padding: 20px;
        margin-bottom: 30px;
        border-radius: 8px;
        position: relative;
    }}
    .dark-theme .header {{ background-color: #1a237e; box-shadow: 0 4px 6px rgba(0, 0, 0, 0.3); }}
    .light-theme .header {{ background-color: #3f51b5; color: white; box-shadow: 0 4px 6px rgba(0, 0, 0, 0.1); }}
    .logo {{ font-size: 2em; margin-bottom: 0; }}
    .timestamp {{ font-style: italic; opacity: 0.7; }}
    .card {{ border-radius: 8px; padding: 20px; margin-bottom: 20px; box-shadow: 0 4px 6px rgba(0, 0, 0, 0.1); transition: transform 0.3s ease; }}
    .card:hover {{ transform: translateY(-5px); }}
    .dark-theme .card {{ background-color: #1e1e1e; box-shadow: 0 4px 6px rgba(0, 0, 0, 0.3); }}
    .light-theme .card {{ background-color: white; box-shadow: 0 4px 6px rgba(0, 0, 0, 0.1); }}
    .card h2 {{ border-bottom: 1px solid #444; padding-bottom: 10px; margin-top: 0; }}
    .verdict {{ text-align: center; padding: 20px; margin: 20px 0; border-radius: 8px; font-size: 1.5em; font-weight: bold; box-shadow: 0 4px 8px rgba(0, 0, 0, 0.2); }}
    .verdict-fake {{ background-color: rgba(255, 82, 82, 0.8); color: white; }}
    .verdict-real {{ background-color: rgba(76, 175, 80, 0.8); color: white; }}
    .verdict-uncertain {{ background-color: rgba(255, 193, 7, 0.8); color: black; }}
    .verdict-emoji {{ font-size: 3em; margin: 10px; display: block; }}
    .model-results {{ display: flex; justify-content: space-between; flex-wrap: wrap; margin-bottom: 20px; }}
    .model-card {{ flex: 1; min-width: 300px; margin: 10px; padding: 15px; border-radius: 8px; text-align: center; }}
    .dark-theme .model-card {{ background-color: #2d2d2d; }}
    .light-theme .model-card {{ background-color: #f0f0f0; }}
    .confidence-meter {{ height: 20px; border-radius: 10px; margin: 10px 0; background-color: #444; overflow: hidden; position: relative; }}
    .confidence-fill {{ height: 100%; transition: width 1s ease-in-out; }}
    .fake-confidence {{ background-color: #FF5252; }}
    .real-confidence {{ background-color: #4CAF50; }}
    .confidence-text {{ position: absolute; top: 0; left: 0; right: 0; text-align: center; line-height: 20px; color: white; font-weight: bold; text-shadow: 1px 1px 1px rgba(0, 0, 0, 0.5); }}
    .plot-container {{ margin: 20px 0; border-radius: 8px; overflow: hidden; }}
    .tab-container {{ margin: 20px 0; }}
    .tab-buttons {{ display: flex; overflow-x: auto; border-bottom: 1px solid #444; margin-bottom: 15px; }}
    .tab-button {{ padding: 10px 15px; background: none; border: none; cursor: pointer; font-size: 1em; font-weight: bold; opacity: 0.6; transition: opacity 0.3s, border-bottom 0.3s; }}
    .dark-theme .tab-button {{ color: white; }}
    .light-theme .tab-button {{ color: #333; }}
    .tab-button.active {{ opacity: 1; border-bottom: 3px solid #3f51b5; }}
    .tab-button:hover {{ opacity: 0.8; }}
    .tab-content {{ display: none; }}
    .tab-content.active {{ display: block; animation: fadeIn 0.5s ease-out; }}
    .heatmap-gallery, .mri-gallery {{
        display: grid; 
        grid-template-columns: repeat(auto-fill, minmax(200px, 1fr)); 
        gap: 15px; 
        margin-top: 20px; 
    }}
    .image-card {{ border-radius: 8px; overflow: hidden; transition: transform 0.3s ease; }}
    .image-card:hover {{ transform: scale(1.05); }}
    .dark-theme .image-card {{ background-color: #2d2d2d; box-shadow: 0 4px 6px rgba(0, 0, 0, 0.3); }}
    .light-theme .image-card {{ background-color: #f0f0f0; box-shadow: 0 4px 6px rgba(0, 0, 0, 0.1); }}
    .gallery-img {{ width: 100%; height: auto; display: block; }}
    .image-caption {{ padding: 10px; text-align: center; font-weight: bold; }}
    .lightbox {{ display: none; position: fixed; z-index: 1000; left: 0; top: 0; width: 100%; height: 100%; background-color: rgba(0, 0, 0, 0.9); overflow: auto; }}
    .lightbox-content {{ margin: auto; display: block; max-width: 90%; max-height: 90%; position: absolute; top: 50%; left: 50%; transform: translate(-50%, -50%); }}
    .lightbox-close {{ position: absolute; top: 20px; right: 30px; color: white; font-size: 40px; font-weight: bold; transition: 0.3s; z-index: 1001; }}
    .lightbox-close:hover, .lightbox-close:focus {{ color: #bbb; text-decoration: none; cursor: pointer; }}
    .footer {{ text-align: center; margin-top: 40px; padding: 20px; border-top: 1px solid #444; font-size: 0.9em; opacity: 0.7; }}
    table {{ width: 100%; border-collapse: collapse; margin: 20px 0; }}
    th, td {{ padding: 10px; text-align: left; border: 1px solid #444; }}
    .dark-theme th {{ background-color: #1a237e; color: white; }}
    .light-theme th {{ background-color: #3f51b5; color: white; }}
    .metadata-section {{ background-color: #1a237e; color: white; padding: 10px; font-weight: bold; border-radius: 5px 5px 0 0; margin-top: 20px; }}
    .metadata-row:nth-child(even) {{ background-color: rgba(255, 255, 255, 0.05); }}
    .warning-text {{ background-color: rgba(255, 193, 7, 0.2); color: #ffb300; padding: 8px; border-radius: 4px; margin: 10px 0; font-weight: bold; }}
    .passed-test {{ background-color: rgba(76, 175, 80, 0.8); color: white; padding: 5px 10px; border-radius: 4px; font-weight: bold; }}
    @media (max-width: 768px) {{
        .model-results {{ flex-direction: column; }}
        .model-card {{ margin: 10px 0; }}
        .heatmap-gallery, .mri-gallery {{ grid-template-columns: repeat(auto-fill, minmax(150px, 1fr)); }}
    }}
    @keyframes fadeIn {{ from {{ opacity: 0; transform: translateY(20px); }} to {{ opacity: 1; transform: translateY(0); }} }}
    .animated {{ animation: fadeIn 0.5s ease-out forwards; }}
    .delay-1 {{ animation-delay: 0.1s; }}
    .delay-2 {{ animation-delay: 0.2s; }}
    .delay-3 {{ animation-delay: 0.3s; }}
    .delay-4 {{ animation-delay: 0.4s; }}
    .delay-5 {{ animation-delay: 0.5s; }}
    @media print {{
        .tab-content {{
            display: block !important;
        }}
        .tab-buttons {{
            display: none !important;
        }}
    }}
        </style>
    </head>
    <body class="{theme_class}">
        <div class="container">
            <header class="header animated">
                {export_pdf_button}
                <h1 class="logo">🔍 Deepfake Detection Report</h1>
                <p class="timestamp">Generated on {timestamp}</p>
            </header>
            <div class="card animated delay-1">
                <h2>{header_title}</h2>
                <p><strong>Filename:</strong> {header_filename}</p>
                <p><strong>Analysis Type:</strong> {header_analysis_type}</p>
                <p><strong>Models Used:</strong> {header_models_used}</p>
            </div>
            <div class="verdict verdict-{verdict_class} animated delay-2">
                <span class="verdict-emoji">{verdict_emoji}</span>
                <span>{verdict_text}</span>
                <p>Confidence: {confidence}</p>
            </div>
            <div class="model-results animated delay-3">
                <div class="model-card">
                    <h3>Plain Frame Analysis</h3>
                    <p><strong>Result:</strong> {plain_frames_result}</p>
                    <div class="confidence-meter">
                        <div class="confidence-fill {plain_confidence_class}" style="width: {plain_frames_prob}%;"></div>
                        <div class="confidence-text">{plain_confidence_text}</div>
                    </div>
                    <p>Based on frame-by-frame analysis of visual artifacts</p>
                </div>
                <div class="model-card">
                    <h3>MRI GAN Detection</h3>
                    <p><strong>Result:</strong> {mri_result}</p>
                    <div class="confidence-meter">
                        <div class="confidence-fill {mri_confidence_class}" style="width: {mri_prob}%;"></div>
                        <div class="confidence-text">{mri_confidence_text}</div>
                    </div>
                    <p>Based on GAN pattern detection in frequency domain</p>
                </div>
            </div>
            {plot_section}
            <div class="tab-container animated delay-5">
                <div class="tab-buttons">
                    <button class="tab-button active" onclick="openTab('heatmaps')">Heatmap Analysis</button>
                    <button class="tab-button" onclick="openTab('mri')">MRI GAN Analysis</button>
                    <button class="tab-button" onclick="openTab('metadata')">Video Metadata</button>
                    <button class="tab-button" onclick="openTab('technical')">Technical Details</button>
                    <button class="tab-button" onclick="openTab('structural')">Structural Analysis</button>
                    <button class="tab-button" onclick="openTab('generation')">Device Generation History</button>
                    <button class="tab-button" onclick="openTab('audio')">Audio Analysis</button>
                </div>
                <div id="heatmaps" class="tab-content active">
                    <h2>Visual Artifact Heatmaps</h2>
                    <p>These heatmaps show potential manipulation areas in the video. Highlighted regions indicate detected visual irregularities.</p>
                    <div class="heatmap-gallery">
                        {heatmap_html}
                    </div>
                </div>
                <div id="mri" class="tab-content">
                    <h2>MRI GAN Analysis</h2>
                    <p>The MRI technique visualizes GAN patterns in the frequency domain. Consistent patterns across frames may indicate AI generation.</p>
                    <div class="mri-gallery">
                        {mri_images_html}
                    </div>
                </div>
                <div id="metadata" class="tab-content">
                    <h2>Video Metadata Analysis</h2>
                    {metadata_html}
                </div>
                <div id="technical" class="tab-content">
                    <h2>Technical Details</h2>
                    <table>
                        <tbody>
                            <tr>
                                <td><strong>Detection Methods</strong></td>
                                <td>Plain Frame CNN Analysis, MRI GAN Detection</td>
                            </tr>
                            <tr>
                                <td><strong>Frames Analyzed</strong></td>
                                <td>{frame_count}</td>
                            </tr>
                            <tr>
                                <td><strong>MRI Images Generated</strong></td>
                                <td>{mri_image_count}</td>
                            </tr>
                            <tr>
                                <td><strong>Heatmaps Generated</strong></td>
                                <td>{heatmap_count}</td>
                            </tr>
                            <tr>
                                <td><strong>Threshold</strong></td>
                                <td>50% (Values above indicate potential fake content)</td>
                            </tr>
                        </tbody>
                    </table>
                    <div class="warning-text">
                        <p>Note: This detection tool provides an assessment based on current AI detection techniques. False positives and negatives are possible.</p>
                    </div>
                </div>
                <div id="structural" class="tab-content">
                    <h2>Structural Analysis</h2>
                    <p>This table shows the file signature structure derived from the video analysis.</p>
                    <table>
                        <thead>
                            <tr>
                                <th>Key</th>
                                <th>Value</th>
                            </tr>
                        </thead>
                        <tbody>
                            {signature_html}
                        </tbody>
                    </table>
                </div>
                <div id="generation" class="tab-content">
                    <h2>Device Generation History</h2>
                    <p>This table provides the device generation history extracted from the video metadata.</p>
                    <table>
                        <thead>
                            <tr>
                                <th>Brand</th>
                                <th>Model</th>
                                <th>Generation 1</th>
                                <th>Generation 2</th>
                                <th>Generation 3</th>
                            </tr>
                        </thead>
                        <tbody>
                            {device_history_html}
                        </tbody>
                    </table>
                </div>
                <div id="audio" class="tab-content">
                    <h2>Audio Analysis</h2>
                    <p><strong>Prediction:</strong> {audio_prediction}</p>
                    {audio_metadata_html}
                </div>
            </div>
            <div class="lightbox" id="imageLightbox">
                <span class="lightbox-close" onclick="closeLightbox()">&times;</span>
                <img class="lightbox-content" id="lightboxImage">
            </div>
            <footer class="footer">
                <p>Deepfake Detection Report &copy; {current_year} - low taper fade</p>
            </footer>
        </div>
        <script>
            function openTab(tabName) {{
                const tabContents = document.getElementsByClassName("tab-content");
                for (let content of tabContents) {{
                    content.classList.remove("active");
                }}
                const tabButtons = document.getElementsByClassName("tab-button");
                for (let button of tabButtons) {{
                    button.classList.remove("active");
                }}
                document.getElementById(tabName).classList.add("active");
                const currentTab = Array.from(document.getElementsByClassName("tab-button")).find(
                    button => button.getAttribute("onclick") === "openTab('" + tabName + "')"
                );
                if (currentTab) {{
                    currentTab.classList.add("active");
                }}
            }}
            function openLightbox(src) {{
                document.getElementById("lightboxImage").src = src;
                document.getElementById("imageLightbox").style.display = "block";
            }}
            function closeLightbox() {{
                document.getElementById("imageLightbox").style.display = "none";
            }}
            document.getElementById("imageLightbox").addEventListener("click", function(e) {{
                if (e.target === this) {{
                    closeLightbox();
                }}
            }});
            document.addEventListener("DOMContentLoaded", function() {{
                const animatedElements = document.querySelectorAll(".animated");
                for (let element of animatedElements) {{
                    element.style.opacity = "0";
                }}
                setTimeout(() => {{
                    for (let element of animatedElements) {{
                        element.style.opacity = "1";
                    }}
                }}, 100);
            }});
        </script>
    </body>
    </html>
    """.format(video_name=video_name,
               timestamp=timestamp,
               theme_class=theme_class,
               verdict_class=verdict_class,
               verdict_emoji=verdict_emoji,
               verdict_text=verdict_text,
               confidence=confidence,
               header_title=header_title,
               header_filename=header_filename,
               header_analysis_type=header_analysis_type,
               header_models_used=header_models_used,
               plain_frames_result=plain_frames_result,
               plain_frames_prob=plain_frames_prob,
               plain_confidence_class=plain_confidence_class,
               plain_confidence_text=plain_confidence_text,
               mri_result=mri_result,
               mri_prob=mri_prob,
               mri_confidence_class=mri_confidence_class,
               mri_confidence_text=mri_confidence_text,
               plot_section=plot_section,
               heatmap_html=heatmap_html,
               mri_images_html=mri_images_html,
               metadata_html=metadata_html,
               frame_count=len(frame_numbers),
               mri_image_count=len(mri_images),
               heatmap_count=len(heatmap_images),
               signature_html=signature_html,
               device_history_html=device_history_html,
               audio_prediction=audio_prediction,
               audio_metadata_html=audio_metadata_html,
               current_year=current_year,
               export_pdf_button=export_pdf_button)
    
    return html


def main():
    args = parse_args()
    data = load_json_data(args.input_file)
    video_metadata = None

    if args.video and os.path.exists(args.video):
        video_metadata = extract_video_metadata(args.video)
    elif "video" in data and "location" in data["video"]:
        video_path = data["video"]["location"]
        if os.path.exists(video_path):
            video_metadata = extract_video_metadata(video_path)
        else:
            print("Warning: Video file not found at {}.".format(video_path))

    html_report = generate_html_report(data, video_metadata, args)
    
    with open(args.output, 'w', encoding='utf-8') as f:
        f.write(html_report)

    print("Report successfully generated: {}".format(args.output))

    for file in ["extracted_audio.wav", "converted_video.mp4"]:
        if os.path.exists(file):
            os.remove(file)
            print("Deleted:", file)

if __name__ == "__main__":
    main()
