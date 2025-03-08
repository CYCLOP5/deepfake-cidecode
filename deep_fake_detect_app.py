import argparse
import os
import cv2
import numpy as np
import torch
import torchvision
import warnings
import multiprocessing
import sys
from tabulate import tabulate
from data_utils.face_detection import *
from deep_fake_detect.utils import *
from deep_fake_detect.DeepFakeDetectModel import *
from data_utils.datasets import *

warnings.filterwarnings("ignore", category=np.VisibleDeprecationWarning)

def overlay_heatmap(image_path, cam):
  
    img = cv2.imread(image_path)
    img = cv2.resize(img, (224, 224))
    cam = cv2.resize(cam, (img.shape[1], img.shape[0]))
    heatmap = cv2.applyColorMap(np.uint8(255 * cam), cv2.COLORMAP_JET)
    output = cv2.addWeighted(img, 0.6, heatmap, 0.4, 0)
    heatmap_path = image_path.replace(".png", "_heatmap.png")
    cv2.imwrite(heatmap_path, output)
    print(f"Saved heatmap: {heatmap_path}")
def image_to_video(image_path, output_video="output.mp4", duration=3, fps=30):
    img = cv2.imread(image_path)
    height, width, _ = img.shape
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    video_writer = cv2.VideoWriter(output_video, fourcc, fps, (width, height))
    frame_count = duration * fps
    for _ in range(frame_count):
        video_writer.write(img)
    video_writer.release()
    return output_video


def predict_deepfake(input_videofile, df_method, debug=True, verbose=False):
    num_workers = multiprocessing.cpu_count() - 2
    model_params = {
        'batch_size': 32,
        'imsize': 224,
        'encoder_name': 'tf_efficientnet_b0_ns'
    }

    prob_threshold_fake = 0.5
    fake_fraction = 0.3

    use_cuda = torch.cuda.is_available()
    device = torch.device("cuda" if use_cuda else "cpu")

    vid = os.path.basename(input_videofile)[:-4]
    output_path = os.path.join("output", vid)
    plain_faces_data_path = os.path.join(output_path, "plain_frames")
    os.makedirs(output_path, exist_ok=True)
    os.makedirs(plain_faces_data_path, exist_ok=True)

    if verbose:
        print(f'Extracting faces from the video')
    extract_landmarks_from_video(input_videofile, output_path, overwrite=True)
    crop_faces_from_video(input_videofile, output_path, plain_faces_data_path, overwrite=True)

    apply_gradcam = False  

    if df_method == 'plain_frames':
        model_path = 'assets/weights/deepplain.chkpt'
        frames_path = plain_faces_data_path
        apply_gradcam = True  
    elif df_method == 'MRI':
        if verbose:
            print(f'Generating MRIs of the faces')
        mri_output = os.path.join(output_path, 'mri')
        predict_mri_using_MRI_GAN(plain_faces_data_path, mri_output, vid, 256, overwrite=True)
        model_path = 'assets/weights/deepmri.chkpt'
        frames_path = mri_output
    else:
        raise Exception("Unknown method")

    if verbose:
        print(f'Detecting DeepFakes using method: {df_method}')
    model = DeepFakeDetectModel(frame_dim=model_params['imsize'], encoder_name=model_params['encoder_name'])
    if verbose:
        print(f'Loading model weights {model_path}')
    check_point_dict = torch.load(model_path)
    model.load_state_dict(check_point_dict['model_state_dict'])
    model = model.to(device)
    model.eval()

    test_transform = torchvision.transforms.Compose([
        torchvision.transforms.Resize((model_params['imsize'], model_params['imsize'])),
        torchvision.transforms.ToTensor(),
        torchvision.transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])

    data_path = os.path.join(frames_path, vid)
    test_dataset = SimpleImageFolder(root=data_path, transforms_=test_transform)
    test_loader = DataLoader(test_dataset, batch_size=model_params['batch_size'], num_workers=num_workers, pin_memory=True)

    if len(test_loader) == 0:
        print('Cannot extract images. Dataloaders empty')
        return None, None, None

    probabilities = []
    all_filenames = []
    all_predicted_labels = []

    cam = None  

    for batch_id, samples in enumerate(test_loader):
        frames = samples[0].to(device)
        frames.requires_grad = True  

        output = model(frames)
        predicted = get_predictions(output).to('cpu').detach().numpy()
        class_probability = get_probability(output).to('cpu').detach().numpy()

        all_predicted_labels.extend(predicted.squeeze())
        probabilities.extend(class_probability.squeeze())
        all_filenames.extend(samples[1])

        if apply_gradcam:  
            output.sum().backward()  
            cam = model.get_gradcam()

        for i, fname in enumerate(samples[1]):
            if apply_gradcam and cam is not None:
                overlay_heatmap(fname, cam[i])  

        total_number_frames = len(probabilities)
        probabilities = np.array(probabilities)

        fake_frames_high_prob = probabilities[probabilities >= prob_threshold_fake]
        number_fake_frames = len(fake_frames_high_prob)
        fake_prob = round(sum(fake_frames_high_prob) / number_fake_frames, 4) if number_fake_frames else 0

        real_frames_high_prob = probabilities[probabilities < prob_threshold_fake]
        number_real_frames = len(real_frames_high_prob)
        real_prob = 1 - round(sum(real_frames_high_prob) / number_real_frames, 4) if number_real_frames else 0
        pred = 1 if fake_prob > real_prob else 0

        if debug:
            print(f'all {probabilities}')
            print(f'real {real_frames_high_prob}')
            print(f'fake {fake_frames_high_prob}')
            print(f"number_fake_frames={number_fake_frames}, number_real_frames={number_real_frames}, total_number_frames={total_number_frames}, fake_fraction={fake_fraction}")
            print(f'fake_prob = {round(fake_prob * 100, 4)}%, real_prob = {round(real_prob * 100, 4)}%  pred={pred}')

        return fake_prob, real_prob, pred

def individual_test():
    print_line()
    debug = True
    verbose = True
    fake_prob, real_prob, pred = predict_deepfake(args.input_videofile, args.method, debug=debug, verbose=verbose)
    if pred is None:
        print_red('Failed to detect DeepFakes')
        return

    label = "REAL" if pred == 0 else "DEEP-FAKE"
    probability = real_prob if pred == 0 else fake_prob
    probability = round(probability * 100, 4)
    
    print_line()
    if pred == 0:
        print_green(f'The video is {label}, probability={probability}%')
    else:
        print_red(f'The video is {label}, probability={probability}%')
    print_line()

import json
import sys
import io

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='DeepFake Detection App')
    parser.add_argument('--input_videofile', action='store', help='Input video file or image')
    args = parser.parse_args()

    log_capture_string = io.StringIO()
    sys.stdout = log_capture_string
    sys.stderr = log_capture_string

    log_messages = []

    try:
        if args.input_videofile is not None:
            if not os.path.isfile(args.input_videofile):
                error_msg = f'Input file not found ({args.input_videofile})'
                print(error_msg)
                log_messages.append(error_msg)
                sys.exit(1)

            file_ext = os.path.splitext(args.input_videofile)[1].lower()
            if file_ext in ['.png', '.jpg', '.jpeg']:
                converted_video = image_to_video(args.input_videofile, "converted_video.mp4")
                print(f'Converted image {args.input_videofile} to video {converted_video}')
                args.input_videofile = converted_video

            print("\nRunning Plain Frames Model...")
            fake_prob_plain, real_prob_plain, pred_plain = predict_deepfake(
                args.input_videofile, 'plain_frames', debug=True, verbose=True
            )

            print("\nRunning MRI Model...")
            fake_prob_mri, real_prob_mri, pred_mri = predict_deepfake(
                args.input_videofile, 'MRI', debug=True, verbose=True
            )

            print("\n--- Results ---")
            print(f"Plain Frames Model -> {'DEEP-FAKE' if pred_plain else 'REAL'} (Fake Prob: {fake_prob_plain * 100:.2f}%)")
            print(f"MRI Model -> {'DEEP-FAKE' if pred_mri else 'REAL'} (Fake Prob: {fake_prob_mri * 100:.2f}%)")

            vid = os.path.basename(args.input_videofile)[:-4]
            plain_frames_location = os.path.join("output", vid, "plain_frames", vid)
            mri_location = os.path.join("output", vid, "mri", vid)

            output = {
                "plain_frames": {
                    "prediction": "DEEP-FAKE" if pred_plain else "REAL",
                    "fake_probability": fake_prob_plain * 100,
                    "real_probability": real_prob_plain * 100,
                    "location": plain_frames_location
                },
                "mri": {
                    "prediction": "DEEP-FAKE" if pred_mri else "REAL",
                    "fake_probability": fake_prob_mri * 100,
                    "real_probability": real_prob_mri * 100,
                    "location": mri_location
                },
                "video": {
                    "name": vid,
                    "location": args.input_videofile
                },
                "log": log_capture_string.getvalue().split("\n") 
            }

            with open("detection_output.json", "w") as json_file:
                json.dump(output, json_file, indent=4)

    except Exception as e:
        print(f"An error occurred: {str(e)}")

    finally:
        sys.stdout = sys.__stdout__
        sys.stderr = sys.__stderr__
