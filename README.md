# Fashion Recommendation System

A content-based image recommendation system that uses deep learning to find visually similar fashion items from the Myntra dataset. Includes both a Jupyter notebook for model training and a Streamlit web app for interactive exploration.

## Overview

This project builds a fashion recommendation engine by extracting visual features from clothing images using a pre-trained ResNet-18 model, then finding similar items using K-Nearest Neighbors (KNN) with cosine similarity.

## Dataset

- **Source:** [Myntra Fashion Product Images (Small)](https://www.kaggle.com/datasets/paramaggarwal/fashion-product-images-small) via Kaggle
- **Format:** JPEG images stored in `myntradataset/images/`
- **Subset used:** 2,000 images (configurable via `SUBSET_SIZE`)

## How It Works

1. **Image Loading** — Scans the dataset directory and loads image paths into a Pandas DataFrame.
2. **Feature Extraction** — Each image is passed through a pre-trained ResNet-18 (with the classification head removed) to produce a 512-dimensional feature vector.
3. **Indexing** — All feature vectors are indexed using `sklearn`'s `NearestNeighbors` with cosine distance.
4. **Recommendation** — Given a query image, the system retrieves the top-6 most visually similar items from the dataset.
5. **Visualization** — Results are displayed in a grid showing the query item alongside its matches and similarity scores.
6. **Persistence** — The extracted features, metadata, and model are saved to disk for reuse.

## Project Structure

```
├── fashion-recommendation.ipynb   # Model training notebook
├── fashion_app.py                 # Streamlit web application
├── myntradataset/
│   ├── images/                    # Fashion product images (.jpg)
│   └── subset_metadata.csv        # Metadata for the image subset
└── models/
    ├── fashion_features.npy       # Extracted feature vectors
    ├── fashion_metadata.csv       # DataFrame metadata
    └── fashion_recommender.pkl    # Serialized model bundle (required by app)
```

## Requirements

Install dependencies with:

```bash
pip install torch torchvision numpy pandas matplotlib seaborn scikit-learn pillow joblib streamlit gdown
```

## Configuration

At the top of the notebook, two flags control performance vs. coverage:

| Variable      | Default | Description                              |
|---------------|---------|------------------------------------------|
| `USE_SUBSET`  | `True`  | Use a random subset of the full dataset  |
| `SUBSET_SIZE` | `2000`  | Number of images in the subset           |
| `BATCH_SIZE`  | `32`    | Images processed per batch               |

Set `USE_SUBSET = False` to run on the full dataset (slower).

## Usage

1. Download the dataset from Kaggle and place images in `myntradataset/images/`.
2. Open `fashion-recommendation.ipynb` in Jupyter.
3. Run all cells top to bottom.
4. The notebook will display recommendation grids and save the trained model to `models/`.

To reuse a saved model:

```python
import joblib
model_data = joblib.load('models/fashion_recommender.pkl')
features_array = model_data['features']
fashion_df     = model_data['metadata']
```

## Streamlit Web App (`fashion_app.py`)

The app provides an interactive UI with three modes, accessible from the sidebar:

### Browse Catalog
Displays a random sample of 10 items from the dataset. Click **Select** on any item to instantly see its top visually similar matches with cosine similarity scores. Use **Shuffle** to load a new sample.

### Upload Image
Upload any fashion image (JPG/PNG) and the app extracts its features on-the-fly using ResNet-18, then returns the closest matches from the catalog.

### Analytics
Shows dataset statistics (total items, unique images, feature dimensions) and a scrollable preview of the metadata table.

### Running the App

**Step 1 — Train the model** (run the notebook first to generate `models/fashion_recommender.pkl`):
```bash
jupyter nbconvert --to notebook --execute fashion-recommendation.ipynb
```

**Step 2 — Launch the app:**
```bash
streamlit run fashion_app.py
```

The app will open at `http://localhost:8501`.

### Dataset Auto-Download (Google Drive)

On first launch, the app automatically downloads the dataset ZIP from Google Drive using `gdown`. To point it at your own hosted dataset, update the file ID in `fashion_app.py`:

```python
GOOGLE_DRIVE_FILE_ID = "your_file_id_here"
```

Make sure the Google Drive file is shared publicly ("Anyone with the link can view"). The download is cached — subsequent launches skip it if `myntradataset/images/` already exists.

> **Note:** The dataset ZIP is ~3 GB. First launch may take several minutes depending on your connection.

## Tech Stack

| Library        | Purpose                              |
|----------------|--------------------------------------|
| PyTorch        | Feature extraction via ResNet-18     |
| torchvision    | Pre-trained models & transforms      |
| scikit-learn   | KNN indexing & cosine similarity     |
| Pandas / NumPy | Data manipulation                    |
| Matplotlib / Seaborn | Visualization                 |
| Pillow         | Image loading                        |
| joblib         | Model serialization                  |
| Streamlit      | Interactive web application          |
| gdown          | Google Drive dataset download        |

## Notes

- The notebook uses `weights=None` for ResNet-18, meaning the model runs with random weights unless you load pre-trained weights manually. For best results, update to `weights=models.ResNet18_Weights.DEFAULT`.
- Random seeds (`42`) are set for reproducibility across NumPy and PyTorch.
- Failed image loads are automatically skipped and removed from the DataFrame.
