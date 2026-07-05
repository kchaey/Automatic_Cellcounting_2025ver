import cv2
import numpy as np
import matplotlib.pyplot as plt
from tkinter import Tk, filedialog, messagebox


# Function to show a message before selecting an image
def prompt_image_selection(message):
    root = Tk()
    root.withdraw()
    messagebox.showinfo("Image Selection", message)
    root.destroy()


# Function to open file dialog and select an image
def select_image(title="Select Image"):
    root = Tk()
    root.withdraw()  # Hide the main window
    file_path = filedialog.askopenfilename(
        title=title,
        filetypes=[("All Image Files", "*.tif *.png *.jpg *.jpeg *.bmp"), 
                   ("TIFF Files", "*.tif"), 
                   ("PNG Files", "*.png"), 
                   ("JPEG Files", "*.jpg;*.jpeg"), 
                   ("Bitmap Files", "*.bmp"), 
                   ("All Files", "*.*")]
    )
    return file_path


# Function to load an image in grayscale
def load_image(image_path):
    image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    if image is None:
        print(f"Error loading image: {image_path}")
    return image


# Function to enhance contrast with CLAHE
def apply_clahe(image):
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    return clahe.apply(image)


# Function to normalize image for display
def normalize_for_display(image):
    return cv2.normalize(image, None, 0, 255, cv2.NORM_MINMAX)


# Function to detect keypoints and compute descriptors using SIFT
def detect_features(image):
    image = apply_clahe(image)  # Apply CLAHE before detection
    sift = cv2.SIFT_create()
    keypoints, descriptors = sift.detectAndCompute(image, None)
    return keypoints, descriptors


# Function to match features using FLANN
def match_features(descriptors1, descriptors2):
    if descriptors1 is None or descriptors2 is None:
        print("❌ One or both descriptor sets are None. Cannot match features.")
        return []

    FLANN_INDEX_KDTREE = 1
    index_params = dict(algorithm=FLANN_INDEX_KDTREE, trees=5)
    search_params = dict(checks=50)
    flann = cv2.FlannBasedMatcher(index_params, search_params)

    descriptors1 = descriptors1.astype(np.float32)
    descriptors2 = descriptors2.astype(np.float32)

    matches = flann.knnMatch(descriptors1, descriptors2, k=2)
    good_matches = [m for m, n in matches if m.distance < 0.7 * n.distance]
    return good_matches


# Function to compute homography
def compute_homography(keypoints1, keypoints2, good_matches):
    if len(good_matches) > 4:
        src_pts = np.float32([keypoints1[m.queryIdx].pt for m in good_matches]).reshape(-1, 1, 2)
        dst_pts = np.float32([keypoints2[m.trainIdx].pt for m in good_matches]).reshape(-1, 1, 2)
        H, mask = cv2.findHomography(src_pts, dst_pts, cv2.RANSAC, 5.0)
        return H
    else:
        print("Not enough good matches found to compute homography!")
        return None


# Function to visualize SIFT feature matching
def visualize_feature_matching(roi_img, keypoints1, whole_brain_img, keypoints2, good_matches):
    if not good_matches:
        print("⚠️ No good matches to visualize.")
        return
    result_img = cv2.drawMatches(roi_img, keypoints1, whole_brain_img, keypoints2, good_matches[:50], None, 
                                 flags=cv2.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS)
    plt.figure(figsize=(10, 5))
    plt.imshow(result_img, cmap='gray')
    plt.title("SIFT Feature Matching (ROI 1 & Whole Brain)")
    plt.axis("off")
    plt.show(block=True)


# Function to visualize ROI 1 alignment (Whole Brain Image with ROI 1 Corner Markers)
def visualize_roi_alignment(whole_brain_img, transformed_corners):
    if transformed_corners is None:
        print("⚠️ Cannot visualize ROI alignment: no transformed corners available.")
        return
    whole_brain_img_color = cv2.cvtColor(normalize_for_display(whole_brain_img), cv2.COLOR_GRAY2BGR)
    print("\n🔹 Transformed ROI 1 Corner Coordinates (in Whole-Brain Image):")
    for i in range(4):
        pt = tuple(map(int, transformed_corners[i][0]))
        print(f"Corner {i+1}: {pt}")
        cv2.circle(whole_brain_img_color, pt, 10, (255, 0, 0), -1)
    plt.figure(figsize=(10, 6))
    plt.imshow(cv2.cvtColor(whole_brain_img_color, cv2.COLOR_BGR2RGB))
    plt.title("Whole-Brain Image with ROI 1 Corner Markers")
    plt.axis("off")
    plt.show(block=True)


# Function to apply homography transformation to ROI Image 2
def align_roi_image(roi_img_2, H, whole_brain_img):
    if H is not None:
        h, w = whole_brain_img.shape  # Use target shape
        aligned_roi_2 = cv2.warpPerspective(roi_img_2, H, (w, h))
        return normalize_for_display(aligned_roi_2)
    else:
        print("Homography matrix not found. Cannot align ROI Image 2.")
        return None


# Function to visualize final aligned ROI 2 image
def visualize_final_alignment(whole_brain_img, aligned_roi_2):
    if aligned_roi_2 is None:
        print("⚠️ Cannot visualize final alignment: aligned ROI 2 is None.")
        return
    overlay = cv2.addWeighted(
        cv2.cvtColor(normalize_for_display(whole_brain_img), cv2.COLOR_GRAY2BGR), 0.7,
        cv2.cvtColor(aligned_roi_2, cv2.COLOR_GRAY2BGR), 0.3,
        0
    )
    overlay_norm = cv2.normalize(overlay, None, 0, 255, cv2.NORM_MINMAX)
    plt.figure(figsize=(12, 6))
    plt.imshow(cv2.cvtColor(overlay_norm.astype(np.uint8), cv2.COLOR_BGR2RGB))
    plt.title("Whole-Brain Image with Aligned ROI 2 (Different Stain)")
    plt.axis("off")
    plt.show(block=True)


# New: Function to blend aligned ROI on top of whole brain and return composite

def overlay_aligned_roi_on_wholebrain(whole_brain_img, aligned_roi_2, alpha=0.3):
    whole_rgb = cv2.cvtColor(normalize_for_display(whole_brain_img), cv2.COLOR_GRAY2BGR)
    aligned_rgb = cv2.cvtColor(normalize_for_display(aligned_roi_2), cv2.COLOR_GRAY2BGR)
    overlay = cv2.addWeighted(whole_rgb, 1 - alpha, aligned_rgb, alpha, 0)
    return overlay
