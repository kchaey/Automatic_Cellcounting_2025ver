import os
import cv2
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from matplotlib.widgets import Slider
from matplotlib.patches import Polygon
from matplotlib.path import Path
import matplotlib.patheffects as path_effects
from sklearn.neighbors import KDTree
from tkinter import Tk, filedialog
from tkinter import simpledialog
from scipy.optimize import linear_sum_assignment


# -----------------------------------------------------------------------------
# USER PARAMETERS
# -----------------------------------------------------------------------------

MATCH_DIST = 12      # px: max distance for co-expression
TRI_AREA   = 60*100/2  # desired triangle area in overlay pixels


#150x150 for bla, il 
#60*100 
#230x70 dhpc 
#75x300 vhpc 
# -----------------------------------------------------------------------------
def draw_free_polygon_roi(overlay_img, roi_img, H, default_points=7):
    """
    Display a popup to let user input number of ROI points, then collect clicks.
    Returns N×2 array of ROI image coordinates.
    """
    num_points = simpledialog.askinteger(
        title="ROI Point Count",
        prompt="How many points do you want to select for the ROI polygon?",
        initialvalue=default_points,
        minvalue=3,
        maxvalue=20
    )

    if not num_points:
        raise RuntimeError("No ROI point count entered.")

    fig, axes = plt.subplots(1, 2, figsize = (12, 6))
    axes[0].imshow(cv2.cvtColor(overlay_img, cv2.COLOR_BGR2RGB))
    axes[0].set_title("Overlay image")
    axes[0].axis('off')

    axes[1].imshow(cv2.cvtColor(roi_img, cv2.COLOR_BGR2RGB))
    axes[1].set_title("Draw polygon on PV ROI image")
    axes[1].axis('off')

    plt.tight_layout()
    pts_roi = np.array(plt.ginput(num_points, timeout=-1))  # 사용자가 오른쪽에서 클릭
    plt.close(fig)

    # Homography로 overlay 좌표계로 변환
    pts_overlay = cv2.perspectiveTransform(
        pts_roi.astype(np.float32).reshape(-1, 1, 2),
        H
    ).reshape(-1, 2)

    return pts_roi, pts_overlay
    
def select_folder(prompt):
    print(prompt)
    path = filedialog.askdirectory()
    if not path:
        raise RuntimeError("No folder selected.")
    return path

def get_prefixes(input_dir):
    """
    Only derive prefixes from the overlay PNG(s) in the input tree.
    Each overlay file named PREFIX_overlay.png will yield one PREFIX.
    """
    prefixes = []
    for root, _, files in os.walk(input_dir):
        for f in files:
            if f.lower().endswith("_overlay.png"):
                prefixes.append(os.path.splitext(f)[0][:-len("_overlay")])
    return sorted(set(prefixes))

#prefix = 115_bla_1_1
def gather_set_files(prefix, input_dir):
    """
    For a given prefix (e.g. '115_dhpc_1_1_10x_d'), find one of each:
      • homography:    prefix + anything + .npy  (must contain "to_wholebrain")
      • overlay image: prefix + "_overlay".png/.jpeg
      • cFos CSV:      prefix + anything + "_c_" + anything + ".csv"
      • PV  CSV:       prefix + anything + "_p_" + anything + ".csv"
      • cFos ROI:      prefix + anything + "_c_" + anything + ".jpg"
      • PV  ROI:       prefix + anything + "_p_" + anything + ".jpg"
    Skips files in deeper subfolders if they don’t start with prefix.
    """
    homog = overlay = csv_c = csv_p = roi_c = roi_p = None

    for root, _, files in os.walk(input_dir):
        for f in files:
            if not f.startswith(prefix):
                continue
            p = os.path.join(root, f)

            # homography: must contain "to_wholebrain" and end in .npy
            if "to_wholebrain" in f and f.lower().endswith(".npy"):
                homog = p

            # overlay image
            elif f.lower().endswith(("_overlay.png","_overlay.jpeg")):
                overlay = p

            # cell_info CSVs
            elif f.lower().endswith(".csv"):
                if "_c_" in f:
                    csv_c = p
                elif "_p_" in f:
                    csv_p = p

            # 4) ROI JPEGs for 10× cFos / PV
            elif f.lower().endswith(".jpg") and "10x" in f.lower():
                if "_10x_c" in f.lower() and "_sub" not in f.lower():
                    roi_c = p
                elif "_10x_p" in f.lower():
                    roi_p = p


                    

    missing = []
    if not homog:   missing.append("homography (*.npy with 'to_wholebrain')")
    if not overlay: missing.append("overlay image (*_overlay.png/.jpeg)")
    if not csv_c:   missing.append("cFos CSV (*.csv with '_c_')")
    if not csv_p:   missing.append("PV  CSV (*.csv with '_p_')")
    if not roi_c:   missing.append("cFos ROI (*.jpg with '_c_')")
    if not roi_p:   missing.append("PV  ROI (*.jpg with '_p_')")

    if missing:
        raise RuntimeError(f"Set '{prefix}' missing: {', '.join(missing)}")

    return homog, overlay, csv_c, csv_p, roi_c, roi_p

def get_corners(cx, cy, w, h, angle_deg):
    a   = np.deg2rad(angle_deg)
    rel = np.array([[-w/2,-h/2],[ w/2,-h/2],[w/2,h/2],[-w/2,h/2]])
    R   = np.array([[np.cos(a), -np.sin(a)],[np.sin(a), np.cos(a)]])
    return rel.dot(R.T) + np.array([cx, cy])


def compute_polygon_area(pts):
    """Shoelace formula for polygon area. pts is an N×2 array."""
    x = pts[:,0]
    y = pts[:,1]
    # wrap-around trick
    return 0.5 * abs(np.dot(x, np.roll(y,-1)) - np.dot(y, np.roll(x,-1)))

def draw_free_triangle_roi(img_roi):
    """
    Let user click exactly 3 points on the ROI image.
    Returns the raw 3×2 array of vertices in ROI‐image coords.
    """
    fig, ax = plt.subplots()
    ax.imshow(cv2.cvtColor(img_roi, cv2.COLOR_BGR2RGB))
    ax.set_title("Click exactly 3 points on ROI image\nClose when done")
    pts_roi = np.array(plt.ginput(3, timeout=-1))
    plt.close(fig)
    return pts_roi  # no scaling

def compute_polygon_area(pts):
    """
    Shoelace formula for polygon area.
    pts: N×2 array, ordered vertices.
    """
    x, y = pts[:,0], pts[:,1]
    return 0.5 * abs(
        np.dot(x, np.roll(y, -1)) -
        np.dot(y, np.roll(x, -1))
    )
       
def load_and_transform(csv_path, H):
    df  = pd.read_csv(csv_path)
    pts = df[['x','y']].to_numpy().astype(np.float32).reshape(-1,1,2)
    wb  = cv2.perspectiveTransform(pts, H).reshape(-1,2)
    df['wb_x'], df['wb_y'] = wb[:,0], wb[:,1]
    return df

def filter_in_box(df, corners_wb):
    path = Path(corners_wb)
    pts  = df[['wb_x','wb_y']].to_numpy()
    return df[path.contains_points(pts)].copy()

def match_coexp_hungarian(df_cfos, df_pv, max_dist=MATCH_DIST):
    """
    One-to-one pairing via Hungarian algorithm.
    Builds the full cFos⇄PV distance matrix, masks out >max_dist,
    then finds the minimal assignment and only keeps pairs under threshold.
    Returns:
      cfos_matched: DataFrame of matched cFos rows
      pv_matched:   DataFrame of matched PV  rows (unique)
    """
    Xc = df_cfos[['wb_x','wb_y']].to_numpy()
    Xp = df_pv[['wb_x','wb_y']].to_numpy()

    # full distance matrix
    D = np.linalg.norm(Xc[:,None,:] - Xp[None,:,:], axis=2)

    # mask distances > max_dist as a large cost
    large = max_dist * 10
    D_masked = D.copy()
    D_masked[D_masked > max_dist] = large

    # solve assignment
    row_idx, col_idx = linear_sum_assignment(D_masked)

    # keep only true pairs
    keep = D_masked[row_idx, col_idx] <= max_dist
    row_idx = row_idx[keep]
    col_idx = col_idx[keep]

    cfos_matched = df_cfos.iloc[row_idx].copy()
    pv_matched   = df_pv.iloc[col_idx].copy()

    return cfos_matched, pv_matched


def map_overlay_box_to_roi(corners_wb, H):
    Hinv = np.linalg.inv(H)
    pts  = corners_wb.astype(np.float32).reshape(-1,1,2)
    roi  = cv2.perspectiveTransform(pts, Hinv).reshape(-1,2)
    return roi

def plot_roi_with_triangle_and_labels(roi_img, tri_verts, df, out_path, title, H):
    """
    roi_img: your ROI‐level image (e.g. cfos)
    tri_verts: 3×2 array of overlay‐space triangle vertices
    df: DataFrame of cells in ROI coords (x,y,label)
    out_path: where to save PNG
    title: figure title
    H: the homography from ROI→whole‐brain
    """
    # Map the triangle from overlay→ROI coordinates
    tri_roi = cv2.perspectiveTransform(
        tri_verts.astype(np.float32).reshape(-1,1,2), 
        np.linalg.inv(H)
    ).reshape(-1,2)

    fig, ax = plt.subplots(figsize=(6,6))
    ax.imshow(cv2.cvtColor(roi_img, cv2.COLOR_BGR2RGB))
    ax.axis('off')
    ax.set_title(title)

    # draw triangle
    loop = np.vstack([tri_roi, tri_roi[0]])
    ax.plot(loop[:,0], loop[:,1], color='yellow', linewidth=2)

    # label the cells in df (already in ROI coords)
    for _, r in df.iterrows():
        txt = ax.text(r['x'], r['y'], str(int(r['label'])),
                      ha='center', va='center', color='red', fontsize=6)
        txt.set_path_effects([path_effects.withStroke(linewidth=0.5,
                                                      foreground='white')])

    fig.savefig(out_path, dpi=300, bbox_inches='tight')
    plt.close(fig)
    
def plot_overlay_with_coexp(roi_c, roi_p, co_df, out_path):
    rgb = np.zeros((*roi_c.shape[:2],3),dtype=np.uint8)
    rgb[...,1] = cv2.cvtColor(roi_c, cv2.COLOR_BGR2GRAY)
    rgb[...,0] = cv2.cvtColor(roi_p, cv2.COLOR_BGR2GRAY)
    fig, ax = plt.subplots(figsize=(6,6))
    ax.imshow(rgb); ax.axis('off')
    for _, r in co_df.iterrows():
        ax.plot(r['x'], r['y'], 'wo', markersize=4)
    fig.savefig(out_path, dpi=300, bbox_inches='tight')
    plt.close(fig)

def debug_roi_polygon_overlay_alignment(overlay_img, roi_img, H, poly_roi_pts, prefix, out_sub):
    """
    polygon ROI points: N×2 in ROI image coordinates
    - overlays the transformed polygon onto both overlay image and ROI image
    - saves side-by-side PNG to out_sub
    """
    poly_wb = cv2.perspectiveTransform(
        poly_roi_pts.astype(np.float32).reshape(-1,1,2),
        H
    ).reshape(-1,2)

    overlay_img_vis = overlay_img.copy()
    roi_img_vis     = roi_img.copy()

    # draw yellow polygon on overlay image
    loop_overlay = np.vstack([poly_wb, poly_wb[0]])
    for i in range(len(poly_wb)):
        pt1 = tuple(loop_overlay[i].astype(int))
        pt2 = tuple(loop_overlay[i+1].astype(int))
        cv2.line(overlay_img_vis, pt1, pt2, (0,255,255), 2)

    # draw red polygon on ROI image
    loop_roi = np.vstack([poly_roi_pts, poly_roi_pts[0]])
    for i in range(len(poly_roi_pts)):
        pt1 = tuple(loop_roi[i].astype(int))
        pt2 = tuple(loop_roi[i+1].astype(int))
        cv2.line(roi_img_vis, pt1, pt2, (0,0,255), 2)

    # show side-by-side
    fig, axes = plt.subplots(1, 2, figsize=(12, 6))
    axes[0].imshow(cv2.cvtColor(overlay_img_vis, cv2.COLOR_BGR2RGB))
    axes[0].set_title("Overlay Image (yellow polygon)")
    axes[0].axis("off")

    axes[1].imshow(cv2.cvtColor(roi_img_vis, cv2.COLOR_BGR2RGB))
    axes[1].set_title("ROI Image (red polygon)")
    axes[1].axis("off")

    plt.tight_layout()
    plt.show()
    out_path = os.path.join(out_sub, f"{prefix}_debug_polygon_alignment.png")
    fig.savefig(out_path, dpi=300, bbox_inches='tight')
    print("  Saved polygon alignment PNG:", out_path)
    
    plt.close(fig)


if __name__=="__main__":
    Tk().withdraw()
    in_dir  = select_folder("1) Select INPUT folder containing all sets.")
    out_dir = select_folder("2) Select OUTPUT folder for results.")

    prefixes = get_prefixes(in_dir)
    for prefix in prefixes:
        try:
            Hf, ovf, csv_c, csv_p, roi_c, roi_p = gather_set_files(prefix, in_dir)
        except RuntimeError as e:
            print(f"Skipping {prefix}: {e}")
            continue

        H  = np.load(Hf)
        ov = cv2.imread(ovf)
        # new triangle:
        # load the PV ROI image (where user draws)
        roi_p_img = cv2.imread(roi_p)

        # 1) User draws free triangle in ROI coords
        # 1) Ask user for number of ROI polygon points (per prefix)
        print(f'\n Processing: {prefix}')
        if 'bla' in prefix.lower():
            default_points = 3  
        if 'hpc' or 'il' in prefix.lower():
            default_points = 3  
        else:
            default_points = 5  # fallback default

        tri_roi, tri_wb = draw_free_polygon_roi(ov, roi_p_img, H, default_points=default_points)



        #tri_roi = draw_free_polygon_roi(roi_p_img, default_points = 3)

        # 3) Compute and log the actual overlay area
        area_overlay = compute_polygon_area(tri_wb)
        print(f"▶ Slide {prefix}: ROI overlay‐space area = {area_overlay:.1f} px²")

        # 1) Load & warp your cell tables
        df_c = load_and_transform(csv_c, H)
        df_p = load_and_transform(csv_p, H)

        # 2) Now you can filter by that triangle
        in_c = filter_in_box(df_c, tri_wb)
        in_p = filter_in_box(df_p, tri_wb)

        # 3) Perform your Hungarian co-exp matching
        cfos_coexp, pv_coexp = match_coexp_hungarian(in_c, in_p, MATCH_DIST)
        co = cfos_coexp

        # filter by triangle in overlay‐space
        in_c, in_p = filter_in_box(df_c, tri_wb), filter_in_box(df_p, tri_wb)

        # one‐to‐one coexp via Hungarian
        cfos_coexp, pv_coexp = match_coexp_hungarian(in_c, in_p, MATCH_DIST)
        co = cfos_coexp

        # create output subfolder
        out_sub = os.path.join(out_dir, prefix)
        os.makedirs(out_sub, exist_ok=True)
        #out_sub = out_dir
        

        # 7) **Add the area as a new column** in each CSV
        for df in (in_c, in_p, co):
            df['ROI_area_px2'] = area_overlay

        # 8) Save the per‐set CSVs
        out_c = os.path.join(out_sub, f"{prefix}_cfos_in_box.csv")
        out_p = os.path.join(out_sub, f"{prefix}_pv_in_box.csv")
        out_co= os.path.join(out_sub, f"{prefix}_coexp_in_box.csv")
        in_c.to_csv(out_c,  index=False)
        in_p.to_csv(out_p,  index=False)
        co .to_csv(out_co, index=False)
        print("  Saved CSVs:", out_c, out_p, out_co)
    

        # plot boxed ROI images
        img_c    = cv2.imread(roi_c)
        img_p    = cv2.imread(roi_p)
        out_cpng = os.path.join(out_sub, f"{prefix}_cfos_boxed.png")
        out_ppng = os.path.join(out_sub, f"{prefix}_pv_boxed.png")
        
        # 디버깅용 polygon alignment 확인 이미지 저장
        debug_roi_polygon_overlay_alignment(
            overlay_img=ov,
            roi_img=img_p,
            H=H,
            poly_roi_pts=tri_roi,
            prefix=prefix,
            out_sub=out_sub
        )

        # new triangle plotting
        plot_roi_with_triangle_and_labels(
            img_c,
            tri_wb,              # <-- your 3×2 triangle verts in whole‐brain overlay space
            in_c,
            out_cpng,
            f"{prefix} cFos",
            H
        )
        plot_roi_with_triangle_and_labels(
            img_p, tri_wb, in_p, out_ppng, f"{prefix} PV", H
        )
        print("  Saved PNGs:", out_cpng, out_ppng)


        # plot co-expression overlay
        out_co_png = os.path.join(out_sub, f"{prefix}_coexp_overlay.png")
        plot_overlay_with_coexp(img_c, img_p, co, out_co_png)
        print("  Saved PNG:", out_co_png)

    print("\n✅ Done! Results in:", out_dir)
