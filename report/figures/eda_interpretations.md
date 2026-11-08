# Exploratory Data Analysis (EDA) Interpretations

### Figure 1: Sample Grids (`sample_grid_ai.png` & `sample_grid_real.png`)
Inspection of the sample grids reveals distinct visual characteristics between the two domains. The AI-generated images (`AiArtData`) exhibit noticeable synthetic texture smoothing, digital painterly lighting, surreal compositions, and characteristic stylistic diffusion artifacts. In contrast, the real images (`RealArt`) comprise authentic organic textures, varied natural illumination, photographic sensor noise, and structured indoor and outdoor scenes.

### Figure 2: Width & Height Distributions (`hist_width.png` & `hist_height.png`)
The resolution histograms indicate substantial dimensional variability across both classes. While both categories feature popular standard square resolutions (e.g., 512x512 and 1024x1024 common to diffusion model outputs), real images exhibit a broader spread of arbitrary rectangular aspect ratios resulting from diverse web crops and cameras. Resizing all inputs to a uniform dimension (224x224) via bilinear interpolation standardizes spatial scales while preserving dominant structural features.

### Figure 3: File Size Distribution (`hist_filesize.png`)
File sizes range from small compressed thumbnails (~50 KB) to high-resolution assets (>1 MB). AI-generated artworks demonstrate a slightly higher median file size due to complex high-frequency synthetic brushwork and uncompressed PNG encodings. To prevent models from learning trivial shortcuts based on raw file metadata or compression artifacts, the training pipeline operates strictly on decoded raw RGB pixel arrays.
