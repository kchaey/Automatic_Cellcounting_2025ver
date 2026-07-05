import os
import re
import numpy as np
import cv2
from roi2wholebrain import (
    load_image,
    detect_features,
    match_features,
    compute_homography,
    align_roi_image,
    overlay_aligned_roi_on_wholebrain
)
from tkinter import filedialog, Tk
from tkinter.messagebox import askyesno

def find_all_roi10_dapi_images(folder):
    return [os.path.join(folder, f)
            for f in os.listdir(folder)
            if "10x" in f.lower() and ("_d." in f.lower() or "_d_" in f.lower())
            and f.lower().endswith((".jpg", ".png", ".tif", ".tiff"))]

def find_associated_images(roi10_dapi_path):
    in_dir = os.path.dirname(roi10_dapi_path)
    base   = os.path.splitext(os.path.basename(roi10_dapi_path))[0]
    parts  = base.split('_')

    if len(parts) < 5:
        raise RuntimeError("Filename doesn’t split into expected parts: " + base)

    mag, stain = None, None
    for i, p in enumerate(parts):
        if p.lower() in ['2x', '4x', '10x']:
            mag = p.lower()
            stain = parts[i + 1].lower() if i + 1 < len(parts) else None
            break

    if not mag or not stain:
        raise RuntimeError("Could not extract mag/stain from filename: " + base)

    prefix_parts = parts[:i]
    found = {
        'roi10_dapi': roi10_dapi_path,
        'middle4_dapi': None,
        'whole2_dapi': None,
        'other10': []
    }

    for fname in os.listdir(in_dir):
        name, _ = os.path.splitext(fname)
        segs = name.split('_')
        if segs[:len(prefix_parts)] != prefix_parts:
            continue

        for j, p in enumerate(segs):
            if p.lower() in ['2x', '4x', '10x']:
                this_mag = p.lower()
                this_stain = segs[j + 1].lower() if j + 1 < len(segs) else None
                fullp = os.path.join(in_dir, fname)
                if this_mag == '10x' and this_stain == 'd':
                
                    found['roi10_dapi'] = fullp
                elif this_mag == '4x' and this_stain == 'd':
                    found['middle4_dapi'] = fullp
                elif this_mag == '2x' and this_stain == 'd':
                    found['whole2_dapi'] = fullp
                elif this_mag == '10x' and this_stain != 'd' and fname.lower().endswith(('.jpg', '.png', '.tif', '.tiff')):
                    found['other10'].append(fullp)
                break
    return found

pattern = re.compile(r'_(\d{2})(?=[_.])')
'''
def strip_two_digit_suffix(filename: str) -> str:
    return pattern.sub('', filename)'''


def strip_after_stain(filename: str) -> str:
    """
    Remove everything after _d, _c, _p unless '_sub' or 'Simple Segmentation' is present.
    Works for all file types (including .h5).
    """
    name, ext = os.path.splitext(filename)
    # flags for preserving suffixes
    keep_sub = '_sub' in name.lower()
    keep_seg = 'simple segmentation' in name.lower()

    parts = name.split('_')
    stain_tags = {'d', 'c', 'p'}

    # flags for preserving suffixes
    keep_sub = '_sub' in name.lower()
    keep_seg = 'simple segmentation' in name.lower()

    parts = name.split('_')
    stain_tags = {'d', 'c', 'p'}

    new_parts = []
    found_stain = False
    for i, part in enumerate(parts):
        new_parts.append(part)
        if part.lower() in stain_tags:
            found_stain = True
            if keep_sub:
                new_parts.append('sub')
            break  
         
    if found_stain:
        final_name = '_'.join(new_parts)
        if keep_seg:
            final_name += '_Simple Segmentation'
        return final_name + ext
    else:
        return filename


def auto_rename_files_in_folder(folder):
    for root, _, files in os.walk(folder):
        for fname in files:
            new_name = strip_after_stain(fname)
            if new_name != fname:
                old_path = os.path.join(root, fname)
                new_path = os.path.join(root, new_name)
                if os.path.exists(new_path):
                    print(f"⚠️  Skipping {fname!r}: target {new_name!r} already exists.")
                else:
                    os.rename(old_path, new_path)
                    print(f"Renamed: {fname!r} → {new_name!r}")
    print("\n✅ Done renaming files.\n")

# ================================================
# MAIN
# ================================================
if __name__ == "__main__":
    Tk().withdraw()
    input_dir = filedialog.askdirectory(title="Select folder with raw images")
    if not input_dir:
        print("❌ No folder selected. Exiting.")
        exit()
        
    if askyesno("Rename Files", "Do you want to automatically rename files (remove _NN suffix)?"):
        auto_rename_files_in_folder(input_dir)
    else:
        print("⚠️ Rename skipped.\n")

    roi_list = find_all_roi10_dapi_images(input_dir)
    if not roi_list:
        print("❌ No 10x DAPI images found.")
        exit()

    print(f"✅ Found {len(roi_list)} 10x DAPI images.\n")
    
    for roi_path in roi_list:
        print(f"processing: {os.path.basename(roi_path)}")
        try:
            imgs = find_associated_images(roi_path)
            imgs['other10'] = list(set(imgs['other10']))  # ✅ 중복 제거
            if not imgs['whole2_dapi']:
                print("❌ No 2x DAPI image. Skipping.")
                continue


    		
            use_middle = bool(imgs['middle4_dapi'])  # 자동 중간 사용 (팝업 X)
    		
            roi_img_1       = load_image(imgs['roi10_dapi'])
            whole_brain_img = load_image(imgs['whole2_dapi'])
    		
            if use_middle:
                middle_img = load_image(imgs['middle4_dapi'])
                kp1, desc1 = detect_features(roi_img_1)
                kp2, desc2 = detect_features(middle_img)
                matches_12 = match_features(desc1, desc2)
                H_10_to_4 = compute_homography(kp1, kp2, matches_12)
                kp3, desc3 = detect_features(middle_img)
                kp4, desc4 = detect_features(whole_brain_img)
                matches_23 = match_features(desc3, desc4)
                H_4_to_2 = compute_homography(kp3, kp4, matches_23)
                
                H = H_4_to_2 @ H_10_to_4
            
            else:
                kp1, desc1 = detect_features(roi_img_1)
                kp2, desc2 = detect_features(whole_brain_img)
                matches = match_features(desc1, desc2)
                H = compute_homography(kp1, kp2, matches)
            if H is None:
                print("❌ Failed to compute homography.")
                continue
            hpath = os.path.splitext(imgs['roi10_dapi'])[0] + "_to_wholebrain_H.npy"
            np.save(hpath, H)
            print("✅ Homography saved:", hpath)
            
            if not imgs['other10']:
                print("⚠️  No matching 10x stain found for overlay.")
                continue
            for other_path in imgs['other10']:
                roi_img_2 = load_image(other_path)
                aligned = align_roi_image(roi_img_2, H, whole_brain_img)
                base2 = os.path.splitext(os.path.basename(imgs['other10'][0]))[0]
                prefix = "_".join(base2.split("_")[:4])
                overlay = overlay_aligned_roi_on_wholebrain(whole_brain_img, aligned)
                out_path = os.path.join(input_dir, f"{prefix}_overlay.png")
                cv2.imwrite(out_path, overlay)
                print("✅ Overlay saved:", out_path)
        except Exception as e:
            print(f"❌ Error: {e}")
    		
				
