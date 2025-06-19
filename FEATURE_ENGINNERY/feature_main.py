# streamlit_feature_driver.py

import streamlit as st
import pandas as pd
import numpy as np
import cv2
import glob
import os
import logging
from multiprocessing import Pool, cpu_count

# Import your modular feature extractors
from glcm_module import feature_extractor as glcm_extractor
from glrlm_module import glrlm_features as glrlm_extractor
from dwt_module import dwt_features
from gabor_module import gt_features
from fos_module import fos
from hog_module import hog_features
from histogram_features import extract_from_image as hist_extractor
from multi_histogram_features import extract_from_image as multi_hist_extractor

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    filename="feature_extraction.log",
    filemode="w",
    format="%(asctime)s - %(levelname)s - %(message)s"
)

# --- Feature Extraction per Image ---
def extract_all_features(img_path):
    try:
        img = cv2.imread(img_path)
        img = cv2.resize(img, (512, 512))
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

        all_features = []
        all_labels = []

        # GLCM
        glcm_df = glcm_extractor([gray])
        all_features.extend(glcm_df.iloc[0].values)
        all_labels.extend(glcm_df.columns)

        # GLRLM
        glrlm_feat = glrlm_extractor(gray)
        all_features.extend(glrlm_feat)
        all_labels.extend([
            "ShortRunEmphasis", "LongRunEmphasis", "GrayLevelNonUniformity",
            "RunLengthNonUniformity", "RunPercentage", "LowGrayLevelRunEmphasis",
            "HighGrayLevelRunEmphasis", "ShortLowGrayLevelEmphasis",
            "ShortRunHighGrayLevelEmphasis", "LongRunLowGrayLevelEmphasis",
            "LongRunHighGrayLevelEmphasis"
        ])

        # DWT
        dwt_feat, dwt_labels = dwt_features(gray, None)
        all_features.extend(dwt_feat)
        all_labels.extend(dwt_labels)

        # Gabor
        gabor_feat, gabor_labels = gt_features(gray, None)
        all_features.extend(gabor_feat)
        all_labels.extend(gabor_labels)

        # FOS
        fos_feat, fos_labels = fos(gray, None)
        all_features.extend(fos_feat)
        all_labels.extend(fos_labels)

        # HOG
        hog_feat, hog_labels = hog_features(gray)
        all_features.extend(hog_feat)
        all_labels.extend(hog_labels)

        # Histogram
        hist_feat, hist_labels = hist_extractor(gray)
        all_features.extend(hist_feat)
        all_labels.extend(hist_labels)

        # Multi-Histogram
        multi_feat, multi_labels = multi_hist_extractor(gray)
        all_features.extend(multi_feat)
        all_labels.extend(multi_labels)

        return img_path, all_features, all_labels

    except Exception as e:
        logging.error(f"Failed on {img_path}: {e}")
        return img_path, [], []

# --- Streamlit App ---
def main():
    st.set_page_config(page_title="Feature Extraction Dashboard", layout="wide")
    st.title("🧠 Medical Image Feature Extractor")
    st.markdown("This app extracts **8 categories of features** using multiprocessing for performance.")

    img_dir = st.text_input("Enter Image Folder Path", "data/test")
    run_btn = st.button("🔍 Run Feature Extraction")

    if run_btn:
        image_paths = glob.glob(os.path.join(img_dir, "**", "*.png"), recursive=True)

        if not image_paths:
            st.error("No .png images found in the given path!")
            return

        st.success(f"Found {len(image_paths)} images. Starting feature extraction...")

        with Pool(cpu_count()) as pool:
            results = list(stqdm(pool.imap(extract_all_features, image_paths), total=len(image_paths)))

        feature_list = [r[1] for r in results if r[1]]
        labels = results[0][2] if results[0][2] else []
        img_paths = [r[0] for r in results if r[1]]

        df = pd.DataFrame(feature_list, columns=labels)
        df["image_path"] = img_paths

        st.dataframe(df.head())
        csv = df.to_csv(index=False).encode("utf-8")
        st.download_button("📥 Download CSV", csv, "extracted_features.csv", "text/csv")

        st.success("Feature extraction completed!")
        logging.info("Feature extraction completed for all images.")

# Show tqdm bar in Streamlit
from streamlit.runtime.scriptrunner import add_script_run_ctx
from tqdm import tqdm

def stqdm(iterable, **kwargs):
    progress = st.progress(0)
    status_text = st.empty()
    total = kwargs.get("total", len(iterable))
    for i, x in enumerate(iterable):
        yield x
        progress.progress((i + 1) / total)
        status_text.text(f"Progress: {i + 1}/{total}")
    progress.empty()
    status_text.text("Done.")

if __name__ == "__main__":
    main()
