import os
import re
import pandas as pd
from tkinter import Tk, filedialog

FNAME_RE = re.compile(
    r'^'
    r'(?P<animal>\d+)_'
    r'(?P<region>[A-Za-z0-9]+)_'
    r'(?P<section1>\d+)_'
    r'(?P<section2>\d+)_'
    r'(?P<stain>cfos|pv|coexp)'
    r'_in_box\.csv$',
    re.IGNORECASE
)

def parse_filename(fname):
    m = FNAME_RE.match(fname)
    if not m:
        return None
    folder_id = f"{m.group('animal')}_{m.group('section1')}_{m.group('section2')}"
    return folder_id, m.group('region').lower(), m.group('stain').lower()

def main():
    Tk().withdraw()
    top = filedialog.askdirectory(title="→ Select the folder containing subfolders with *_in_box.csv")
    if not top:
        print("❌ No folder selected.")
        return

    summary = {}  # {(folder_id, region): {cfos, pv, coexp}}

    for root, _, files in os.walk(top):
        for fname in files:
            if not fname.endswith("_in_box.csv"):
                continue
            parsed = parse_filename(fname)
            if not parsed:
                print(f"⚠️ Skipping unrecognized file: {fname}")
                continue
            folder_id, region, stain = parsed
            path = os.path.join(root, fname)
            try:
                df = pd.read_csv(path)
                count = len(df)
                area = df['ROI_area_px2'].mean() if 'ROI_area_px2' in df.columns else None
            except Exception as e:
                print(f"⚠️ Failed to load {fname}: {e}")
                continue

            key = (folder_id, region)
            if key not in summary:
                summary[key] = {'cfos': 0, 'pv': 0, 'coexp': 0, 'area_px2': area}
            summary[key][stain] = count

    # Convert to DataFrame
    records = []
    for (folder_id, region), counts in summary.items():
        records.append({
            'Folder': folder_id,
            'Regions': region,
            'cFos': counts.get('cfos', 0),
            'PV': counts.get('pv', 0),
            'Co-expression': counts.get('coexp', 0),
            'Area_px2': counts.get('area_px2')
        })

    df = pd.DataFrame(records, columns=['Folder', 'Regions', 'cFos', 'PV', 'Co-expression', 'Area_px2'])

    # Sort by defined region order
    region_order = ['bla', 'ila', 'dhpc', 'vhpc']
    df['Regions'] = pd.Categorical(df['Regions'], categories=region_order, ordered=True)
    df = df.sort_values(['Regions', 'Folder']).reset_index(drop=True)

    out_path = os.path.join(top, "cell_counts_summary.csv")
    df.to_csv(out_path, index=False)
    print("✅ Summary saved to:", out_path)

if __name__ == "__main__":
    main()

