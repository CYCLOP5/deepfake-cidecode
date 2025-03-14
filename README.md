# DeepFake Detection Pipeline – Technical Overview

This project implements a multi-modal DeepFake detection pipeline that processes video, image, and audio inputs to determine if the media has been manipulated. The system uses advanced deep learning techniques for visual analysis and classical machine learning for audio analysis, integrating two parallel detection methods: a plain frames‑based method and an MRI‑GAN‑based method. Below is an in‑depth technical explanation of the pipeline's logic."

You can read more about the project [here](https://lowtaperfadedeepfake.my.canva.site/).  

---

## **The Workflow of the Project is as shown below:**
![Workflow](https://github.com/CYCLOP5/deepfake-cidecode/blob/ml-stuff/image.png)

## **The Tech Stack of the Project is as shown below:**
![Tech Stack](https://github.com/CYCLOP5/deepfake-cidecode/blob/ml-stuff/image(1).png)
## 1. Input Handling and Preprocessing

### 1.1. Media Input Types
- **Video:**  
  - Frames are extracted at regular intervals (e.g., every 10th frame) to reduce computational load.
- **Image:**  
  - Static images are converted into short video clips by repeating the image for a set duration (e.g., 3 seconds at 30 fps).
- **Audio:**  
  - Audio files are processed using Librosa:
    - **Loading & Trimming:** Audio is loaded at a target sample rate (e.g., 16 kHz) and trimmed of silence using `librosa.effects.trim`.
    - **Normalization & Length Adjustment:** The waveform is normalized (dividing by its maximum absolute value) and padded or truncated to a fixed length (e.g., 32000 samples).
    - **Feature Extraction:** A mel‑spectrogram is computed with 128 mel bands, converted to decibels, and then flattened to serve as input for a logistic regression model.

### 1.2. Face Detection and Extraction
- **Face Detection:**  
  - A pre‑trained face detection module, specifically the Multi‑task Cascaded Convolutional Network (MTCNN), is employed to detect faces in each frame.
  - Following the approach detailed in the original MRI‑GAN paper, frames are sampled (typically every 10th frame) to reduce computational cost. MTCNN is then applied to each selected frame to detect facial landmarks and crop the corresponding facial regions.
  - In cases where MTCNN fails to detect any faces (which can occur due to low‑quality, noisy, or highly distracted frames), those frames or entire videos are dropped from the dataset.
  - **Data Augmentation:**  
    - Prior to face detection, various augmentation and distraction techniques (e.g., rotations, scaling, flipping, and overlaying distractor elements) are applied to simulate real‑world conditions and improve model robustness.

---

## 2. DeepFake Detection Models

The pipeline implements two parallel methods to detect manipulated faces:

### 2.1. Plain Frames‑based Detection
- **Model Architecture:**  
  - Utilizes Efficient‑Net B0, pre‑trained on ImageNet and fine‑tuned for binary classification (Real vs. Fake).
- **Processing Flow:**  
  - Each detected face is resized to 224×224, normalized with standard mean and standard deviation values, and directly fed into the classifier.
- **Grad‑CAM Integration:**  
  - Optionally, Grad‑CAM is applied to generate activation heatmaps, providing visual explanations for the classification decisions.

### 2.2. MRI‑GAN Based Detection
- **MRI‑GAN Architecture:**  
  - **Generator:**  
    - Based on a U‑Net architecture with 16 layers, featuring skip connections between encoder and decoder layers.
    - Down‑sampling modules use convolution, instance normalization, LeakyReLU, and dropout; up‑sampling modules employ transposed convolutions, instance normalization, and ReLU.
  - **Discriminator:**  
    - Implements a PatchGAN architecture that focuses on local image patches, thereby enforcing high‑frequency detail accuracy.
- **Loss Functions:**  
  - **Conditional GAN Loss:** Encourages the generator to produce realistic MRI outputs conditioned on the input face.
  - **Pixel‑wise L2 Loss:** Measures the mean squared error between the generated MRI and the ground truth MRI.
  - **Perceptual (SSIM‑based) Loss:**  
    - Uses a modified Structural Similarity Index Metric (SSIM) to assess perceptual differences.
    - Defined as:  
      ```
      L_per(G) = √(1 - SSIM(x, y))
      ```
- **Loss Aggregation:**  
  - The total generator loss is computed as:
    ```
    L(G) = L_cGAN(G, D) + λ (τ * L_L2(G) + (1 - τ) * L_per(G))
    ```
    where λ is a scaling factor (set to 100) and τ is a hyperparameter that balances the contribution of the pixel‑wise L2 loss and the perceptual loss.
- **Hyperparameter τ:**  
  - In our implementation, **τ is set to 0.15** (instead of 0.3), which places more emphasis on the perceptual loss component during MRI generation.
- **Classification on MRI Images:**  
  - The MRI images generated by the GAN are passed through a dedicated Efficient‑Net B0 classifier to produce a DeepFake prediction.

---

## 3. Audio DeepFake Detection

**Dataset:** [The Fake‑or‑Real (FoR) Dataset](https://www.kaggle.com/datasets/mohammedabdeldayem/the-fake-or-real-dataset)

### **Workflow for Audio Detection:**
- Audio is loaded using Librosa.
- Silent parts of the audio are skipped, and fully silent files are ignored.
- The volume (dB) of all files is normalized.
- All files are converted to the same length to ensure identical feature vectors.
- Audio is converted into mel‑spectrograms, which are then flattened into a 1D array for model training.

### **Dataset Splitting:**
- **Train:** (57072, 8064), Labels: (57072,)
- **Test:** (5722, 8064), Labels: (5722,)
- **Validation:** (10798, 8064), Labels: (10798,)

### **Classifier Models Used:**
- Random Forest Classifier (RFC)
- Logistic Regression
- XGBoost Classifier

**Final Model Choice:** Logistic Regression was selected as the final classifier due to its promising results without overfitting.

---

## 4. Aggregation and Final Decision

- **Face-Level Aggregation:**  
  For each video, predictions from every detected face (from both the plain frames and MRI‑GAN branches) are aggregated.
- **Fake Fraction Threshold:**  
  A predetermined fake fraction threshold is used to determine if the entire video is manipulated.
- **Audio Integration:**  
  If audio is present, its prediction (obtained via a logistic regression model) is incorporated into the final decision.
- **Output Generation:**  
  The aggregated results, including prediction probabilities, Grad‑CAM visualizations, and file locations of processed outputs, are logged and saved as a JSON report.

---

## 5. End-to-End Workflow

- **Input File Identification:**  
  Determine if the input is a video, image, or audio file based on the file extension.
- **Preprocessing:**  
  - **Video:** Extract frames and detect faces.
  - **Image:** Convert to video, then extract frames and faces.
  - **Audio:** Process to extract mel‑spectrogram features.
- **Parallel Detection:**  
  - **Plain Frames Method:** Direct classification of faces using Efficient‑Net B0.
  - **MRI‑GAN Method:** Generate MRI images from faces using a GAN (with τ = 0.15) and classify them using a dedicated Efficient‑Net B0.
  - **Audio Analysis:** Process audio features through a logistic regression model.
- **Aggregation:**  
  Combine face‑level predictions using a fake fraction threshold and merge with audio predictions to form a multi‑modal decision.
- **Result Output:**  
  The final DeepFake decision and detailed logs are outputted in a JSON file.

---

## 6. Dataset and Training Data Generation

The MRI‑GAN model is trained on a custom dataset called the **MRI‑DF dataset**, which is comprised of image pairs: the original face image and its corresponding MRI image. For real faces, the MRI image is simply a blank (black) image, indicating no manipulation. For DeepFake faces, however, the MRI image captures the structural differences between the fake image and its corresponding real source image using a perceptual dissimilarity measure (SSIM).

To construct the MRI‑DF dataset, we followed the methodology described in the original paper (Pratikkumar Prajapati and Chris Pollett, *MRI‑GAN: A Generalized Approach to Detect DeepFakes using Perceptual Image Assessment*, [arXiv:2203.00108](https://arxiv.org/abs/2203.00108)):

- **Data Sources:**  
  - **Fake Samples:**  
    - Randomly selected 50% of the videos from the DeepFake Detection Challenge (DFDC) training set.
    - Included all videos from the Celeb‑DF‑v2 dataset.
  - **Real Samples:**  
    - Augmented the dataset with real face images from the FDF and FFHQ datasets to balance the representation, as the DFDC and Celeb‑DF‑v2 datasets contain a higher proportion of fake content.

- **Mapping Fake to Real:**  
  The DFDC and Celeb‑DF‑v2 datasets provide metadata linking fake video clips to their corresponding real videos. This metadata is crucial for generating accurate MRI images by pairing each fake face with its real counterpart.

- **MRI Generation Process:**  
  - For each fake face, the MRI image is computed by applying an SSIM‑based perceptual dissimilarity function between the fake frame and its corresponding real frame.
  - For real faces, a blank (black) image is used as the MRI.

- **Data Augmentation:**  
  - Various augmentation and distraction techniques (such as rotations, scaling, adding noise, and overlaying textual or geometric distractors) are applied to the DFDC training data prior to computing SSIM.
  - These augmentations help simulate real‑world distortions and improve the robustness of the model.
  - Note that augmentations are not applied to the Celeb‑DF‑v2, FDF, and FFHQ datasets, ensuring their MRI images remain unaltered.

This comprehensive dataset allows the MRI‑GAN to learn a mapping from a face image to its MRI representation—generating a blank output for real faces and an artifact‑rich output for manipulated faces—which is pivotal for accurate DeepFake detection.

---

## 7. Credits

This project builds upon the foundational concepts introduced in the MRI‑GAN model for DeepFake detection by Pratikkumar Prajapati and Chris Pollett. While inspired by their work, we have made several adaptations and optimizations to enhance functionality and tailor the implementation to our specific use case.

**Reference:**  
- Pratikkumar Prajapati, Chris Pollett, *\"MRI‑GAN: A Generalized Approach to Detect DeepFakes using Perceptual Image Assessment\"*, 2022. [arXiv:2203.00108](https://arxiv.org/abs/2203.00108)




