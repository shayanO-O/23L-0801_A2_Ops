# VisionGuard Grad-CAM Qualitative Interpretability Report

## 1. Interpretability Overview
Grad-CAM (Gradient-weighted Class Activation Mapping) computes the gradients of the target class score with respect to the feature activation maps of the final convolutional layer (`out_relu` in MobileNetV2). By spatially pooling these gradients and weighting the feature maps, Grad-CAM highlights which spatial regions the neural network relies upon when reaching a classification verdict.

## 2. Analysis of Correct Predictions (True Positives & True Negatives)
1. **Authentic / Real Images (TN)**:
   - For correctly classified genuine photographs, the activation maps exhibit low, diffuse gradients across organic textures, skin grain, and complex natural lighting.
   - The model verifies high-frequency camera noise and irregular micro-contrasts typical of physical optical lenses and Bayer sensors, preventing false trigger on semantic subjects.
2. **AI-Generated Images (TP)**:
   - For correctly detected synthetic images, Grad-CAM activations localize sharply on over-smoothed skin regions, unnatural specular reflections, and hyper-detailed edge transitions characteristic of latent diffusion upscaling.
   - In synthetic portraits and renders, attention focuses on hair boundaries and iris symmetry anomalies where generative artifacts cluster.

## 3. Failure Mode Diagnosis (False Positives & False Negatives)
1. **False Positives (Real Misclassified as AI)**:
   - In both analyzed false positive instances, the authentic images feature intense studio lighting, heavy post-production color grading, and shallow depth-of-field bokeh.
   - The model's convolutional kernels mistake smooth portrait bokeh backgrounds and airbrushed advertising retouches for generative latent diffusion smoothing.
2. **False Negatives (AI Misclassified as Real)**:
   - For generative images that escaped detection, Grad-CAM reveals that the network focused on realistic photographic composition, low-contrast shadows, or synthetic noise injection that mimics ISO camera grain.
   - Because the generative artifacts in these samples were subtle and global rather than localized, the localized spatial filters in `out_relu` failed to register decisive activation peaks above $\tau = 0.5252$.
