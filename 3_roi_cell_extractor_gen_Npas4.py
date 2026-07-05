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
from scipy.optimize import linear_sum_assignment
from tkinter import simpledialog

# -----------------------------------------------------------------------------
# USER PARAMETERS
# -----------------------------------------------------------------------------
MIN_AREA   = 30      # px²: drop any tiny blobs
MATCH_DIST = 12     # px: max distance for co-expression
#BOX_W      = 230     # overlay-pixel width of your fixed box
#BOX_H      = 70    # overlay-pixel height of your fixed box

#60*100 IL  (너무 가운데(안쪽으로 하지 말기))
#230x70 dhpc 
#70x230 vhpc 
#150x150 for bla, il 
#60*100 
#230x70 dhpc 
#75x300 vhpc 
# -----------------------------------------------------------------------------

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
                if "_10x_c" in f.lower():
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

def draw_fixed_box(img):
    fig, ax = plt.subplots()
    ax.imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
    ax.set_title(
        "Click to move box center; use slider to rotate.\n"
        "Close when done."
    )
    mgr = plt.get_current_fig_manager()
    try: mgr.toolbar.show()
    except: mgr.toolbar_visible = True

    cy0, cx0 = img.shape[0]/2, img.shape[1]/2
    box = {'cx':cx0,'cy':cy0,'w':BOX_W,'h':BOX_H,'angle':0.}

    corners = get_corners(cx0, cy0, BOX_W, BOX_H, 0)
    patch = Polygon(corners, closed=True, edgecolor='yellow', fill=False, linewidth=2)
    ax.add_patch(patch)

    def onclick(evt):
        if evt.inaxes!=ax or evt.button!=1: return
        box['cx'], box['cy'] = evt.xdata, evt.ydata
        pts = get_corners(box['cx'], box['cy'], BOX_W, BOX_H, box['angle'])
        patch.set_xy(pts); fig.canvas.draw_idle()
    fig.canvas.mpl_connect('button_press_event', onclick)

    s_ax   = fig.add_axes([0.2,0.02,0.6,0.04])
    slider = Slider(s_ax, "Angle°", -180, 180, valinit=0.)
    def onrotate(val):
        box['angle']=val
        pts = get_corners(box['cx'], box['cy'], BOX_W, BOX_H, val)
        patch.set_xy(pts); fig.canvas.draw_idle()
    slider.on_changed(onrotate)

    plt.show(block=True)
    return box['cx'], box['cy'], box['w'], box['h'], box['angle']

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

def plot_roi_with_box_and_labels(roi_img, cx, cy, w, h, ang,
                                 df, out_path, title, H):
    fig, ax = plt.subplots(figsize=(6,6))
    ax.imshow(cv2.cvtColor(roi_img, cv2.COLOR_BGR2RGB))
    ax.axis('off')
    ax.set_title(title)

    corners_wb = get_corners(cx, cy, w, h, ang)
    corners_roi= map_overlay_box_to_roi(corners_wb, H)
    loop       = np.vstack([corners_roi, corners_roi[0]])
    ax.plot(loop[:,0], loop[:,1], color='yellow', linewidth=2)

    for _, r in df.iterrows():
        txt = ax.text(r['x'], r['y'], str(int(r['label'])),
                      ha='center', va='center', color='red', fontsize=6)
        txt.set_path_effects([path_effects.withStroke(linewidth=0.5, foreground="white")])

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


if __name__=="__main__":
    Tk().withdraw()
    in_dir  = select_folder("1) Select INPUT folder containing all sets.")
    out_dir = select_folder("2) Select OUTPUT folder for results.")

    prefixes = get_prefixes(in_dir)
    if not prefixes:
        print("❌ No overlay files found. Check for *_overlay.png.")
        exit(1)

    for prefix in prefixes:
        try:
            Hf, ovf, csv_c, csv_p, roi_c, roi_p = gather_set_files(prefix, in_dir)
            	# 이미지 로드
            ov = cv2.imread(ovf)
            img_roi = cv2.imread(roi_c)  # or roi_p
            # 해상도 비율 계산 (ROI가 더 크므로 보통 scale > 1)
            scale_x = img_roi.shape[1] / ov.shape[1]
            scale_y = img_roi.shape[0] / ov.shape[0]
            scale_factor = (scale_x + scale_y) / 2  # 평균값 사용

        except RuntimeError as e:
            print(f"⚠️  Skipping set '{prefix}': {e}")
            continue

        print(f"\n=== Processing set: {prefix} ===")
        H  = np.load(Hf)
        ov = cv2.imread(ovf)
        BOX_W = simpledialog.askinteger("Box Width", "Enter box width (px):", initialvalue=300, minvalue=10, maxvalue=1000)
        BOX_H = simpledialog.askinteger("Box Height", "Enter box height (px):", initialvalue=300, minvalue=10, maxvalue=1000)
        if not BOX_W or not BOX_H: 
        	raise RuntimeError("Box size not entered.")
        	
        # draw & rotate box
        cx, cy, w, h, ang = draw_fixed_box(ov)
        w_roi = w * scale_factor
        h_roi = h * scale_factor
        corners_wb = get_corners(cx, cy, w_roi, h_roi, ang)

        # load & warp cells
        df_c = load_and_transform(csv_c, H)
        df_p = load_and_transform(csv_p, H)

        # filter and co-exp
        in_c = filter_in_box(df_c, corners_wb)
        in_p = filter_in_box(df_p, corners_wb)

# new:
        cfos_coexp, pv_coexp = match_coexp_hungarian(in_c, in_p, MATCH_DIST)
        co = cfos_coexp

        # create output subfolder
        out_sub = os.path.join(out_dir, prefix)
        os.makedirs(out_sub, exist_ok=True)

        # save CSVs
        out_c = os.path.join(out_sub, f"{prefix}_Npas4_in_box.csv")
        out_p = os.path.join(out_sub, f"{prefix}_pv_in_box.csv")
        out_co= os.path.join(out_sub, f"{prefix}_coexp_in_box.csv")
        in_c.to_csv(out_c, index=False)
        in_p.to_csv(out_p, index=False)
        co .to_csv(out_co,index=False)
        print("  Saved CSVs:", out_c, out_p, out_co)

        # plot boxed ROI images
        img_c    = cv2.imread(roi_c)
        img_p    = cv2.imread(roi_p)
        out_cpng = os.path.join(out_sub, f"{prefix}_Npas4_boxed.png")
        out_ppng = os.path.join(out_sub, f"{prefix}_pv_boxed.png")
        plot_roi_with_box_and_labels(img_c, cx, cy, w_roi, h_roi, ang, in_c, out_cpng, f"{prefix} Npas4", H)
        plot_roi_with_box_and_labels(img_p, cx, cy, w_roi, h_roi, ang, in_p, out_ppng, f"{prefix} PV", H)

        print("  Saved PNGs:", out_cpng, out_ppng)

        # plot co-expression overlay
        out_co_png = os.path.join(out_sub, f"{prefix}_coexp_overlay.png")
        plot_overlay_with_coexp(img_c, img_p, co, out_co_png)
        print("  Saved PNG:", out_co_png)

    print("\n✅ Done! Results in:", out_dir)
