import os
import re
from tkinter import Tk, filedialog

def select_folder(prompt):
    print(prompt)
    path = filedialog.askdirectory()
    if not path:
        print("❌ No folder selected. Exiting.")
        exit(1)
    return path

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



if __name__ == "__main__":
    Tk().withdraw()
    top = select_folder("→ Select the folder whose files you want to rename:")
    print(f"Renaming files under: {top}\n")

    for root, _, files in os.walk(top):
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

    print("\n✅ Done! All applicable files have been renamed.")
