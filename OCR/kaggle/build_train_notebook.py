"""Generate vietnamese_receipts_mc_ocr_train.ipynb for Kaggle."""
import json
from pathlib import Path

NB_PATH = Path(__file__).parent / "vietnamese_receipts_mc_ocr_train.ipynb"
PADDLE_COMPAT_SNIPPET = (
    Path(__file__).parent.parent / "src/receipt_ocr/paddle_compat.py"
).read_text(encoding="utf-8")


def md(source: str):
    return {"cell_type": "markdown", "metadata": {}, "source": _cell_source(source)}


def code(source: str):
    return {
        "cell_type": "code",
        "metadata": {},
        "source": _cell_source(source),
        "outputs": [],
        "execution_count": None,
    }


def _cell_source(source: str) -> list[str]:
    """Chia từng dòng thật (không splitlines — tránh cắt literal \\n trong một dòng)."""
    if not source.endswith("\n"):
        source += "\n"
    parts = source.split("\n")
    lines = [p + "\n" for p in parts[:-1]]
    if parts[-1]:
        lines.append(parts[-1] + "\n")
    return lines or ["\n"]


cells = [
    md(
        """# Huấn luyện OCR hóa đơn tiếng Việt (MC-OCR 2021) trên Kaggle

Notebook này huấn luyện pipeline nhận dạng hóa đơn theo hướng:

- **[MC_OCR top-1](https://github.com/ndcuong91/MC_OCR)**: PaddleOCR phát hiện vùng chữ → VietOCR đọc nội dung → trích xuất trường (SELLER, ADDRESS, TIMESTAMP, TOTAL_COST).
- **[invoice_ocr_vietnamese](https://github.com/PakeNguyen/invoice_ocr_vietnamese)**: gom dòng + rule trích xuất tên quán, tổng tiền, ngày.

**Dataset Kaggle (bắt buộc thêm vào notebook):**

`/kaggle/input/datasets/domixi1989/vietnamese-receipts-mc-ocr-2021`

**Cài đặt trước khi chạy:**

1. Settings → Accelerator → **GPU** (T4/P100).
2. Add Data → dataset `vietnamese-receipts-mc-ocr-2021` (domixi1989).
3. Internet **On**.

**Đầu ra:** `/kaggle/working/receipt_ocr_artifacts/` — tải về chạy demo local."""
    ),
    md(
        """## Giai đoạn 0 — Cài đặt thư viện

`vietocr==0.3.13` trên PyPI **ghim** `einops==0.2.0` — không cài `einops` mới hơn. Cài VietOCR trước, PaddleOCR sau.

**Python 3.12+ (Kaggle):** `paddlepaddle==3.2.2` + `paddleocr` 3.x; `enable_mkldnn=False` (tránh lỗi PIR/oneDNN).

PaddleOCR 3.x: dùng `predict(img)`, không dùng `ocr(img, cls=True)`.

Trên Kaggle, Paddle chạy **CPU** (detection); VietOCR vẫn dùng **GPU**."""
    ),
    code(
        """# %% [Giai đoạn 0] Cài đặt
import importlib.util
import subprocess
import sys

def pip(*packages, reinstall=False):
    cmd = [sys.executable, "-m", "pip", "install", "-q"]
    if reinstall:
        cmd.extend(["--force-reinstall", "--no-cache-dir"])
    subprocess.check_call(cmd + list(packages))

# 1) VietOCR (kéo einops==0.2.0, pillow==10.2.0, ...)
pip("vietocr==0.3.13")

# 2–3) Paddle runtime + PaddleOCR (phụ thuộc phiên bản Python)
import os

# Trước import paddle: tránh PIR+oneDNN crash trên CPU (Paddle 3.3+)
os.environ.setdefault("FLAGS_use_mkldnn", "0")
os.environ.setdefault("PADDLE_PDX_DISABLE_MODEL_SOURCE_CHECK", "True")

PADDLE_CPU_INDEX = "https://www.paddlepaddle.org.cn/packages/stable/cpu/"
PY312 = sys.version_info >= (3, 12)


def _uninstall_paddle_wheels():
    subprocess.run(
        [sys.executable, "-m", "pip", "uninstall", "-y", "paddlepaddle", "paddlepaddle-gpu"],
        capture_output=True,
    )


def _install_paddle():
    on_kaggle = os.path.isdir("/kaggle")
    # Python 3.12: chỉ có wheel Paddle 3.x; 2.6.1 không cài được.
    if PY312 or on_kaggle:
        _uninstall_paddle_wheels()
        if PY312:
            pip("setuptools>=68.0.0")
        pip(
            "paddlepaddle==3.2.2",
            "-i",
            PADDLE_CPU_INDEX,
        )
        return
    if importlib.util.find_spec("paddle") is not None:
        return
    try:
        pip(
            "paddlepaddle-gpu==2.6.1",
            "-f",
            "https://www.paddlepaddle.org.cn/whl/linux/mkl/avx/stable.html",
        )
    except subprocess.CalledProcessError:
        pip("paddlepaddle==2.6.1")


def _install_paddleocr():
    if PY312:
        pip("paddleocr>=3.0.0,<4.0.0")
    else:
        pip("paddleocr>=2.7.0,<3.0.0")


def _install_paddlex_extras():
    # PaddleOCR 3.x kéo paddlex → cần langchain_text_splitters (optional dep chưa khai báo đủ)
    if not PY312:
        return
    pip("langchain-text-splitters")
    pip("langchain", "langchain-community")


_install_paddle()
_install_paddleocr()
_install_paddlex_extras()

# 4) Sửa PIL lệch bản (vietocr pin 10.2.0 + paddleocr → file PIL lẫn version)
def _fix_pillow():
    for ver in ("10.4.0", "11.1.0"):
        pip(f"Pillow=={ver}", reinstall=True)
        try:
            import importlib
            import PIL.ImageFont  # noqa: F401
            from PIL import Image
            print("Pillow:", Image.__version__)
            return
        except ImportError:
            pass
    raise RuntimeError("Không sửa được Pillow — restart session và chạy lại cell 0.")

_fix_pillow()


def _patch_numpy_for_imgaug():
    # imgaug 0.4.0 dùng np.sctypes (đã bỏ ở NumPy 2; Kaggle không hạ được numpy<2)
    import numpy as np

    if hasattr(np, "sctypes"):
        return
    np.sctypes = {
        "float": [np.float16, np.float32, np.float64],
        "int": [np.int8, np.int16, np.int32, np.int64],
        "uint": [np.uint8, np.uint16, np.uint32, np.uint64],
        "complex": [np.complex64, np.complex128],
        "others": [np.bool_, object, bytes, str],
    }


_patch_numpy_for_imgaug()
import numpy as np

print("NumPy:", np.__version__)

import torch

print("PyTorch:", torch.__version__, "| CUDA:", torch.cuda.is_available())
if torch.cuda.is_available():
    print("GPU:", torch.cuda.get_device_name(0))

import einops
import vietocr
from vietocr.model.trainer import Trainer  # sau _patch_numpy_for_imgaug()
import paddle
from paddleocr import PaddleOCR

print("einops:", einops.__version__)
print("vietocr:", getattr(vietocr, "__version__", "0.3.13"))
print("Paddle:", paddle.__version__)
print("PaddleOCR + VietOCR Trainer: OK")
# Cảnh báo pip khác từ gói sẵn Kaggle — bỏ qua nếu import OK."""
    ),
    md(
        """## Giai đoạn 1 — Đường dẫn dataset & khám phá dữ liệu (EDA)

Dataset MC-OCR: ~1155 ảnh train, crop dòng chữ, CSV nhãn trường."""
    ),
    code(
        """# %% [Giai đoạn 1] EDA
import os
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd

KAGGLE_ROOT = Path("/kaggle/input/datasets/domixi1989/vietnamese-receipts-mc-ocr-2021")
if not KAGGLE_ROOT.is_dir():
    KAGGLE_ROOT = Path("/kaggle/input/vietnamese-receipts-mc-ocr-2021")
assert KAGGLE_ROOT.is_dir(), (
    "Thêm dataset domixi1989/vietnamese-receipts-mc-ocr-2021 vào notebook"
)


def nested(root: Path, name: str) -> Path:
    direct, inner = root / name, root / name / name
    return inner if inner.is_dir() else direct


TRAIN_IMG_DIR = nested(KAGGLE_ROOT, "train_images")
VAL_IMG_DIR = nested(KAGGLE_ROOT, "val_images")
CROP_DIR = nested(KAGGLE_ROOT, "text_recognition_mcocr_data")
TRAIN_CSV = KAGGLE_ROOT / "mcocr_train_df.csv"
TRAIN_ANN = KAGGLE_ROOT / "text_recognition_train_data.txt"
VAL_ANN = KAGGLE_ROOT / "text_recognition_val_data.txt"

WORK = Path("/kaggle/working")
ARTIFACTS = WORK / "receipt_ocr_artifacts"
ARTIFACTS.mkdir(parents=True, exist_ok=True)

n_train = len(list(TRAIN_IMG_DIR.glob("*.jpg")))
n_val = len(list(VAL_IMG_DIR.glob("*.jpg")))
n_crop = len(list(CROP_DIR.glob("*.jpg")))

print("Dataset root:", KAGGLE_ROOT)
print("Train images:", TRAIN_IMG_DIR, "→", n_train)
print("Val images:", VAL_IMG_DIR, "→", n_val)
print("Text crops:", CROP_DIR, "→", n_crop)
assert n_crop > 0, "Không thấy crop ảnh — kiểm tra đường dẫn text_recognition_mcocr_data"

df = pd.read_csv(TRAIN_CSV)
print("mcocr_train_df:", df.shape)

label_counts = {}
for s in df["anno_labels"].fillna(""):
    for lab in str(s).split("|||"):
        label_counts[lab] = label_counts.get(lab, 0) + 1
print("Top labels:", sorted(label_counts.items(), key=lambda x: -x[1])[:10])

samples = sorted(TRAIN_IMG_DIR.glob("*.jpg"))
assert samples, "Không có ảnh train"
fig, ax = plt.subplots(1, 1, figsize=(6, 8))
ax.imshow(plt.imread(samples[0]))
ax.set_title(samples[0].name)
ax.axis("off")
plt.show()"""
    ),
    md(
        """## Giai đoạn 2 — Chuẩn bị dữ liệu fine-tune VietOCR

VietOCR cần `data_root` chứa ảnh + file annotation. **Không** symlink cả thư mục `images` vào `/kaggle/input` (read-only) — chỉ symlink từng file `.jpg` vào `/kaggle/working`."""
    ),
    code(
        """# %% [Giai đoạn 2] Chuẩn bị annotation VietOCR
import shutil

OCR_DATA = WORK / "vietocr_data"
if OCR_DATA.exists():
    shutil.rmtree(OCR_DATA)
OCR_DATA.mkdir(parents=True)

img_root = OCR_DATA / "images"
img_root.mkdir(parents=True, exist_ok=True)

# Thư mục trong /kaggle/working (ghi được). Symlink từng crop → input read-only.
crop_files = list(CROP_DIR.glob("*.jpg"))
print(f"Liên kết {len(crop_files)} ảnh crop vào {img_root} ...")
linked, copied = 0, 0
for src in crop_files:
    dst = img_root / src.name
    if dst.exists() or dst.is_symlink():
        continue
    try:
        dst.symlink_to(src)
        linked += 1
    except OSError:
        shutil.copy2(src, dst)
        copied += 1
print(f"  symlink: {linked} | copy: {copied}")

assert any(img_root.glob("*.jpg")), f"Thư mục ảnh trống: {img_root}"


def build_ann(src_txt: Path, dst_txt: Path) -> int:
    lines_out = []
    missing = 0
    with open(src_txt, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line or "\t" not in line:
                continue
            fname, label = line.split("\t", 1)
            fname, label = fname.strip(), label.strip()
            if not label:
                continue
            img_path = img_root / fname
            if not img_path.is_file():
                missing += 1
                continue
            label = label.replace("\t", " ")
            lines_out.append(f"{fname}\t{label}")
    dst_txt.parent.mkdir(parents=True, exist_ok=True)
    dst_txt.write_text(chr(10).join(lines_out), encoding="utf-8")
    print(dst_txt.name, "lines:", len(lines_out), "| missing:", missing)
    return len(lines_out)


train_ann = img_root / "train_annotation.txt"
val_ann = img_root / "val_annotation.txt"
n_tr = build_ann(TRAIN_ANN, train_ann)
n_va = build_ann(VAL_ANN, val_ann)
assert n_tr > 0 and n_va > 0, "Annotation rỗng — kiểm tra file txt và thư mục crop"

print("Sample:")
print(train_ann.read_text(encoding="utf-8").splitlines()[:3])"""
    ),
    md(
        """## Giai đoạn 3 — Fine-tune VietOCR (vgg_transformer)

Huấn luyện từ pretrained. LMDB cache tạo tại `/kaggle/working` (đổi cwd trước khi train)."""
    ),
    code(
        """# %% [Giai đoạn 3] Train VietOCR
import os
import shutil
from pathlib import Path

# Shim NumPy 2 cho imgaug (nếu chạy lại cell 3 mà không chạy cell 0)
def _patch_numpy_for_imgaug():
    import numpy as np

    if hasattr(np, "sctypes"):
        return
    np.sctypes = {
        "float": [np.float16, np.float32, np.float64],
        "int": [np.int8, np.int16, np.int32, np.int64],
        "uint": [np.uint8, np.uint16, np.uint32, np.uint64],
        "complex": [np.complex64, np.complex128],
        "others": [np.bool_, object, bytes, str],
    }


_patch_numpy_for_imgaug()

from vietocr.tool.config import Cfg
from vietocr.model.trainer import Trainer

BATCH_SIZE = 16  # giảm nếu OOM trên T4
NUM_ITERS = 8000
VALID_EVERY = 2000
PRINT_EVERY = 200

VIETOCR_DATA_ROOT = str(img_root) + os.sep

config = Cfg.load_config_from_name("vgg_transformer")
config["device"] = "cuda:0" if __import__("torch").cuda.is_available() else "cpu"
config["cnn"]["pretrained"] = False

config["dataset"].update({
    "name": "mcocr_receipt",
    "data_root": VIETOCR_DATA_ROOT,
    "train_annotation": "train_annotation.txt",
    "valid_annotation": "val_annotation.txt",
})

config["trainer"].update({
    "batch_size": BATCH_SIZE,
    "print_every": PRINT_EVERY,
    "valid_every": VALID_EVERY,
    "iters": NUM_ITERS,
    "checkpoint": str(ARTIFACTS / "vietocr_checkpoint.pth"),
    "export": str(ARTIFACTS / "vietocr_receipt.pth"),
    "log": str(ARTIFACTS / "train.log"),
})
config["dataloader"]["num_workers"] = 0  # ổn định trên Kaggle

config_path = ARTIFACTS / "config.yml"
export_path = Path(config["trainer"]["export"])
ckpt_path = Path(config["trainer"]["checkpoint"])

# LMDB tạo trong cwd → chạy từ /kaggle/working
_orig_cwd = os.getcwd()
os.chdir(WORK)
try:
    trainer = Trainer(config, pretrained=True)
    trainer.config.save(str(config_path))
    print("Saved config:", config_path)
    trainer.train()
finally:
    os.chdir(_orig_cwd)

# Luôn có weights export (kể cả khi val acc không tăng)
if not export_path.is_file():
    if ckpt_path.is_file():
        shutil.copy(ckpt_path, export_path)
        print("Copied checkpoint → export:", export_path)
    else:
        trainer.save_weights(str(export_path))
        print("Saved final weights:", export_path)

assert export_path.is_file(), "Không tạo được vietocr_receipt.pth"
print("Export size (MB):", round(export_path.stat().st_size / 1e6, 2))"""
    ),
    md(
        """## Giai đoạn 4 — Đánh giá nhanh trên ảnh validation

PaddleOCR detect + VietOCR đã fine-tune + rule trích xuất trường."""
    ),
    code(
        """# %% [Giai đoạn 4] Đánh giá mẫu
import os
import re
import sys

import cv2
import numpy as np
import pandas as pd
from PIL import Image
from vietocr.tool.config import Cfg
from vietocr.tool.predictor import Predictor

"""
        + PADDLE_COMPAT_SNIPPET
        + """


def extract_receipt_fields(df_lines: pd.DataFrame) -> dict:
    lines = [str(x).strip() for x in df_lines["line_text"].tolist() if str(x).strip()]
    info = {"seller": None, "address": None, "timestamp": None, "total_cost": None}
    if not lines:
        return info
    for line in lines[:5]:
        low = line.lower()
        if "ngày" not in low and "tổng" not in low and "tong" not in low:
            info["seller"] = line
            break
    for line in lines:
        if re.search(r"\\d{1,2}[/-]\\d{1,2}[/-]\\d{2,4}", line):
            info["timestamp"] = line
            break
    amounts = []
    for line in reversed(lines):
        for n in re.findall(r"[\\d.,]+", line):
            d = re.sub(r"\\D", "", n)
            if d and int(d) >= 1000:
                amounts.append(int(d))
    if amounts:
        info["total_cost"] = max(amounts)
    return info


weights = export_path
assert weights.is_file(), f"Thiếu weights: {weights} — chạy lại giai đoạn 3"

cfg = Cfg.load_config_from_file(str(config_path))
cfg["weights"] = str(weights)
cfg["device"] = config["device"]
cfg["cnn"]["pretrained"] = False
predictor = Predictor(cfg)
paddle = init_paddleocr(use_gpu=not os.path.isdir("/kaggle"))


def pipeline_one(img_path):
    bgr = cv2.imread(str(img_path))
    if bgr is None:
        raise FileNotFoundError(img_path)
    img = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
    res = run_paddle_ocr(paddle, img)
    rows = []
    for line in paddle_lines(res):
        box, rec = line
        tp, conf = rec if isinstance(rec, (list, tuple)) else (str(rec), 0.0)
        xs = [p[0] for p in box]
        ys = [p[1] for p in box]
        x1, y1 = int(min(xs)), int(min(ys))
        x2, y2 = int(max(xs)), int(max(ys))
        if x2 <= x1 or y2 <= y1:
            continue
        crop = img[y1:y2, x1:x2]
        try:
            text = predictor.predict(Image.fromarray(crop)) or tp
        except Exception:
            text = tp
        rows.append({"x1": x1, "y1": y1, "x2": x2, "y2": y2, "text": str(text).strip()})
    df = pd.DataFrame(rows)
    if df.empty:
        return {}, df
    df = df.sort_values("y1")
    groups, cur, last_y = [], [], -1e9
    for _, r in df.iterrows():
        yc = (r["y1"] + r["y2"]) / 2
        if abs(yc - last_y) > 30 and cur:
            groups.append(cur)
            cur = []
        cur.append(r.to_dict())
        last_y = yc
    if cur:
        groups.append(cur)
    line_texts = []
    for grp in groups:
        grp = sorted(grp, key=lambda x: x["x1"])
        line_texts.append(" ".join(x["text"] for x in grp if x["text"]))
    df_lines = pd.DataFrame({"line_text": line_texts})
    return extract_receipt_fields(df_lines), df_lines


val_samples = sorted(VAL_IMG_DIR.glob("*.jpg"))[:5]
assert val_samples, "Không có ảnh validation"
for p in val_samples:
    fields, lines = pipeline_one(p)
    print("---", p.name)
    print("Extracted:", fields)
    if not lines.empty:
        print(lines["line_text"].tolist()[:8])"""
    ),
    md(
        """## Giai đoạn 5 — Đóng gói artifact & lưu Kaggle Output

Tải `receipt_ocr_artifacts.zip` về `Train/OCR/models/` để chạy demo."""
    ),
    code(
        """# %% [Giai đoạn 5] Lưu artifact
import json
import zipfile

meta = {
    "dataset": str(KAGGLE_ROOT),
    "train_images": n_train,
    "val_images": n_val,
    "text_crops": n_crop,
    "vietocr_weights": "vietocr_receipt.pth",
    "config": "config.yml",
    "num_iters": NUM_ITERS,
    "batch_size": BATCH_SIZE,
}
(ARTIFACTS / "meta.json").write_text(
    json.dumps(meta, indent=2, ensure_ascii=False), encoding="utf-8"
)

zip_path = WORK / "receipt_ocr_artifacts.zip"
with zipfile.ZipFile(zip_path, "w", zipfile.ZIP_DEFLATED) as zf:
    for f in ARTIFACTS.rglob("*"):
        if f.is_file():
            zf.write(f, f.relative_to(ARTIFACTS.parent))

print("Artifacts in", ARTIFACTS)
for f in sorted(ARTIFACTS.iterdir()):
    if f.is_file():
        print(f"  {f.name}: {f.stat().st_size // 1024} KB")
print("Zip:", zip_path, "→", zip_path.stat().st_size // 1024, "KB")"""
    ),
    md(
        """## Giai đoạn 6 — Demo local (sau khi tải artifact)

```powershell
cd D:\\Luan-Van\\Train
.\\.venv\\Scripts\\Activate.ps1
pip install -r OCR\\requirements.txt
# Giải nén receipt_ocr_artifacts.zip → OCR\\models\\
$env:RECEIPT_OCR_WEIGHTS = "D:\\Luan-Van\\Train\\OCR\\models\\vietocr_receipt.pth"
cd OCR\\demo
uvicorn server:app --reload --host 127.0.0.1 --port 8010
```

Mở http://127.0.0.1:8010"""
    ),
]

notebook = {
    "nbformat": 4,
    "nbformat_minor": 5,
    "metadata": {
        "kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
        "language_info": {"name": "python", "version": "3.10.0"},
        "accelerator": "GPU",
    },
    "cells": cells,
}

NB_PATH.write_text(json.dumps(notebook, ensure_ascii=False, indent=1), encoding="utf-8")
print("Wrote", NB_PATH)
