import subprocess
import sys
import os
from tkinter import Tk, filedialog, simpledialog

# Step 0: asking cell type 
Tk().withdraw()

# choose the Cell Type
cell_type = simpledialog.askstring(
    title="Cell Type 선택",
    prompt="분석할 cell type을 입력하세요 ('cfos' 또는 'npas4'):",
).strip().lower()

if cell_type not in ["cfos", "npas4"]:
    raise ValueError("❌ Cell type must be 'cfos' or 'npas4'")

# Step 0.5: asking roi (ex. bla, il, dhpc)
prefix = simpledialog.askstring(
    title="roi 입력",
    prompt="처리할 이미지 roi 를 입력하세요 (예: bla, il, dhpc):"
).strip().lower()

if not prefix:
    raise RuntimeError("❌ No ROI")

# 현재 실행 중인 Python 환경 사용
python_exe = sys.executable

# Step 1: run 1_roi2wholebrain_up_rename.py
print("\n▶ Step 1: Aligning ROI to Whole Brain...")
subprocess.run([python_exe, "1_roi2wholebrain_up_rename.py"], check=True)

# Step 2: run 2_batch_cell_extractor_simp_seg.py
print("\n▶ Step 2: Extracting cells from segmentation...")
subprocess.run([python_exe, "2_batch_cell_extractor_simp_seg.py"], check=True)

# Step 3: run ROI cell extractor
print(f"\n▶ Step 3: Extracting ROI info for {cell_type}...")

# 선택 논리 추가
if "bla" in prefix:
    script3 = "3_roi_cell_extractor_bla.py"
elif cell_type == "cfos":
    script3 = "3_roi_cell_extractor_gen_matching_drag.py"
else:  # npas4
    script3 = "3_roi_cell_extractor_gen_Npas4.py"

# 실행
subprocess.run([python_exe, script3], check=True)

print("\n✅ All steps completed successfully!")
