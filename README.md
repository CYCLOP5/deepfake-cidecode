# DeepFake Detection Pipeline Logic

This project implements a multi-modal DeepFake detection system that processes video, image, and audio inputs to determine whether the media is manipulated. Below is an overview of the core logic and workflow behind the model.

---

## 1. Input Handling and Preprocessing

- **Media Input Types:**  
  The system accepts video, image, or audio files.
  - **Audio:**  
    - Loaded using Librosa.
    - Trimmed for silence, normalized, and padded/truncated to a fixed length.
    - A mel-spectrogram is computed and converted to decibel scale before being fed into a logistic regression audio model.
  - **Image:**  
    - Converted into a short video by repeating the image frames (e.g., 3 seconds at 30 fps).
  - **Video:**  
    - Frames are extracted (typically every tenth frame) for further processing.

- **Face Detection:**  
  - A pre-trained face detection module extracts facial regions from each frame.
  - This ensures that only the relevant parts (i.e., faces) are processed in subsequent steps.

---

## 2. DeepFake Detection Models

The pipeline employs two parallel deep learning methods for DeepFake detection:

### A. Plain Frames-based Method
- **Direct Classification:**  
  - Each detected face is directly fed into a pre-trained Efficient-Net B0 model.
  - The model extracts features and classifies the face as either "Real" or "Fake."
- **Grad-CAM Integration:**  
  - Optionally, Grad-CAM is applied to generate heatmaps that visualize the regions influential in the decision-making process.

### B. MRI-GAN Based Method
- **MRI Generation:**  
  - Detected faces are first processed through an MRI-GAN generator.
  - **Output:**
    - A blank (black) image for real faces.
    - An image with perceptual artifacts for manipulated (DeepFake) faces.
  - This approach leverages structural similarity (SSIM) to capture subtle differences introduced during face manipulation.
- **MRI-GAN Loss Function:**  
  - The objective function for MRI-GAN combines three components:
    1. Conditional GAN loss.
    2. Pixel-wise L2 loss.
    3. Perceptual (SSIM-based) loss.
  - A hyperparameter **τ (tau)** controls the balance between the L2 loss and the perceptual loss.
  - In this project, **τ is set to 0.15** (instead of 0.3) to improve the sensitivity of the MRI generation.
- **Classification on MRI Images:**  
  - The generated MRI images are classified using a dedicated Efficient-Net B0 model.

---

## 3. Aggregation and Final Decision

- **Frame-Level Aggregation:**  
  - Individual face predictions from both methods are aggregated to determine the overall classification for the video.
- **Fake Fraction Threshold:**  
  - A predetermined fake fraction threshold is used during aggregation to decide if the video is classified as a DeepFake.
- **Audio Integration:**  
  - If audio is available, it is separately processed and its prediction is incorporated into the overall decision.
- **Final Output:**  
  - The aggregated predictions from the plain frames method, MRI-GAN method, and audio analysis are combined.
  - Detailed logs, prediction probabilities, and file locations are saved in a JSON file.

---

## 4. End-to-End Workflow

1. **Input File Identification:**  
   - Determine if the input is video, image, or audio.
2. **Preprocessing:**  
   - Convert and preprocess the media as required (e.g., extract audio, convert image to video, extract faces).
3. **Parallel Detection:**  
   - **Plain Frames Method:** Direct face classification using Efficient-Net B0.
   - **MRI-GAN Method:** Generate MRI images from faces (using τ = 0.15 in the loss function) and classify using a dedicated Efficient-Net B0.
4. **Aggregation:**  
   - Aggregate frame-level predictions using the fake fraction threshold to produce a final video-level DeepFake decision.
5. **Result Output:**  
   - Combine and log the results, including Grad-CAM visualizations (if enabled), into a JSON report.

---

This pipeline leverages advanced deep learning and signal processing techniques alongside a robust software stack (including PyTorch, OpenCV, Librosa, and a React + TypeScript frontend) to provide a comprehensive DeepFake detection solution.






This project incorporates and builds upon the foundational concepts of the MRI-GAN model for DeepFake detection, originally proposed by Pratikkumar Prajapati and Chris Pollett. While inspired by their work, we have adapted and modified aspects of the implementation to enhance its functionality and optimize it for our specific use case.

- Pratikkumar Prajapati, Chris Pollett, *"MRI-GAN: A Generalized Approach to Detect DeepFakes using Perceptual Image Assessment"*, 2022. [arXiv:2203.00108](https://arxiv.org/abs/2203.00108)
