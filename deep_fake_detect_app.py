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
    """
    Overlay Grad-CAM heatmap on the original image.
    """
    img = cv2.imread(image_path)
    img = cv2.resize(img, (224, 224))

    cam = cv2.resize(cam, (img.shape[1], img.shape[0]))

    heatmap = cv2.applyColorMap(np.uint8(255 * cam), cv2.COLORMAP_JET)

    output = cv2.addWeighted(img, 0.6, heatmap, 0.4, 0)
    heatmap_path = image_path.replace(".png", "_heatmap.png")
    cv2.imwrite(heatmap_path, output)
    print(f"Saved heatmap: {heatmap_path}")

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

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='DeepFake Detection App')
    parser.add_argument('--input_videofile', action='store', help='Input video file')
    parser.add_argument('--method', action='store', choices=['plain_frames', 'MRI'], help='Method type')

    args = parser.parse_args()

    if args.input_videofile is not None:
        if args.method is None:
            parser.print_help(sys.stderr)
        else:
            if os.path.isfile(args.input_videofile):
                individual_test()
            else:
                print(f'Input file not found ({args.input_videofile})')
    else:
        parser.print_help(sys.stderr)