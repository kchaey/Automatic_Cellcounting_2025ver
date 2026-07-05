import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.patheffects as path_effects
from skimage.measure import regionprops_table, label
from tkinter import Tk, filedialog
import cv2
import h5py

def convert_h5_to_npy_in_folder(input_dir, output_dir=None):
    if output_dir is None:
        output_dir = input_dir

    for fname in os.listdir(input_dir):
        if fname.endswith(".h5") and "Simple Segmentation" in fname:
            h5_path = os.path.join(input_dir, fname)
            npy_name = fname.replace(".h5", ".npy")
            npy_path = os.path.join(output_dir, npy_name)

            with h5py.File(h5_path, "r") as f:
                key = list(f.keys())[0]  # 일반적으로 'exported_data'
                data = f[key][:]
                np.save(npy_path, data)

            print(f"✅ {fname} → {npy_name}")

def process_segmentation_file(seg_path, output_dir):
    seg = np.load(seg_path).squeeze()
    unique_vals = np.unique(seg)

    if unique_vals.max() > 2:
        labeled = seg.astype(np.uint16)
        print(f"✔ Using existing labels in: {os.path.basename(seg_path)}")
    elif set(unique_vals.tolist()) == {1, 2}:
        # 1 = cell, 2 = background
        labeled = label((seg == 1).astype(np.uint8), connectivity=1)
        print(f"🔁 Relabeled from class-1 cell mask: {os.path.basename(seg_path)}")
    else:
        labeled = label((seg > 0).astype(np.uint8), connectivity=1)
        print(f"⚠️  Relabeled general binary mask: {os.path.basename(seg_path)}")
    
    # cfos / pv 구분
    fname = os.path.basename(seg_path).lower()
    if "_c_" in fname:
        min_area = 35  # 예: cfos
    elif "_p_" in fname:
        min_area = 60  # 예: pv
    else:
        min_area = 45  # 기본값 (필요 시)

    # extract regionprops
    props = regionprops_table(
        labeled,
        properties=['label', 'centroid', 'area']
    )
    df = pd.DataFrame(props)
    df = df[df['area'] > min_area]   # filter noise
    df.rename(columns={'centroid-0': 'y', 'centroid-1': 'x'}, inplace=True)

    base_name = os.path.splitext(os.path.basename(seg_path))[0]
    csv_path  = os.path.join(output_dir, f"{base_name}_cell_info.csv")
    img_path  = os.path.join(output_dir, f"{base_name}_labeled_cells.png")

    df.to_csv(csv_path, index=False)

    # find matching raw image
    raw_prefix = base_name.split('_Simple Segmentation')[0]
    background = None
    for fname in os.listdir(os.path.dirname(seg_path)):
        if fname.lower().endswith(('.jpg', '.png', '.tif', '.jpeg')) \
           and fname.startswith(raw_prefix):
            background = cv2.imread(os.path.join(os.path.dirname(seg_path), fname))
            background = background.astype(np.uint8)
            
            if background is not None:
                print(f"📷 Found matching raw image: {fname}")
                mask = ((labeled > 0).astype(np.uint8) * 255)
                background = np.stack([labeled * 255] * 3, axis=-1)
                background = background.astype(np.uint8)

                break

    if background is None:
        print("⚠️  No raw image found. Using labeled mask as background.")
        background = np.stack([labeled * 255] * 3, axis=-1)

    # plot labels
    fig, ax = plt.subplots(figsize=(10, 10))
    ax.imshow(cv2.cvtColor(background, cv2.COLOR_BGR2RGB))
    for _, row in df.iterrows():
        txt = ax.text(
            row['x'], row['y'], str(int(row['label'])),
            color='red', fontsize=6, ha='center', va='center'
        )
        txt.set_path_effects([
            path_effects.Stroke(linewidth=0.5, foreground='white'),
            path_effects.Normal()
        ])
    ax.set_title(base_name)
    ax.axis('off')
    plt.savefig(img_path, bbox_inches='tight', dpi=300)
    plt.close()

    print(f"📄 Saved CSV:   {csv_path}")
    print(f"🖼️  Saved image: {img_path}\n")


def batch_process_segmentations():
    Tk().withdraw()

    # 1) pick input folder
    print("1) Select the folder containing your Simple Segmentation .h5 or .npy files & raw image files")
    folder_path = filedialog.askdirectory(title="Select segmentation folder")
    if not folder_path:
        print("❌ No segmentation folder selected. Exiting.")
        return
    output_dir = folder_path
    '''
    # 2) pick output folder
    
    print("2) Select (or create) the folder where CSVs and labeled images will be saved.")
    output_dir = filedialog.askdirectory(title="Select or create output folder")
    if not output_dir:
        print("❌ No output folder selected. Exiting.")
        return'''
     

    # convert h5 -> npy if needed
    convert_h5_to_npy_in_folder(folder_path)
    
    # 3) find segmentation files
    files = [f for f in os.listdir(folder_path) if f.endswith("Simple Segmentation.npy")]
    if not files:
        print("❌ No Simple Segmentation .npy files found. Exiting.")
        return
    print(f"🔍 Found {len(files)} segmentation files in:\n   {folder_path}\n")

    # 4) process each
    for fname in files:
        seg_path = os.path.join(folder_path, fname)
        process_segmentation_file(seg_path, output_dir)


if __name__ == "__main__":
    batch_process_segmentations()
