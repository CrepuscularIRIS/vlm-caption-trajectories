"""Caption 裁图环境（protocol v5.1），供 GRPO 多轮 rollout 使用。

呈现尺寸策略由 7,047 张 SFT 视图实测反推：
- obs0(原图)：面积 > 1400²=1.96M 时等比缩小，绝不放大；
- 裁图：短边补到 448（放大上限 4.0x），面积超 1.96M 同样缩小；
- 边长对齐 28 的倍数（Qwen 视觉 2x2 patch merge 对齐）。
文本渲染复用公开参考实现 caption_v51（SYSTEM/parse_step/first_user_text/receipt_text）。
"""
import re
import os  # [scrub]
import sys
from pathlib import Path

from PIL import Image

_REPO = Path(__file__).resolve().parents[3]  # [scrub] repo root
REF = Path(os.environ.get("CAPTION_REF_DIR", _REPO / "code" / "reference"))  # [scrub]
if str(REF) not in sys.path:
    sys.path.insert(0, str(REF)); sys.path.insert(0, str(REF))  # [scrub]  # caption_v3 依赖

import caption_v52i as proto  # noqa: E402  (v5.2i: Step-1 WORD is intent only)

MAX_CROPS = 8  # 交付口径：v4c 预算与首库一致（caption_v6 模块默认 16，不用）
MIN_CROP_PX = 56
AREA_CAP = 1400 * 1400
MIN_SHOWN_SHORT = 448
MAX_UPSCALE = 4.0
ALIGN = 28  # 对齐 SFT 出图规则（28 + 短边448放大，含原图）


def _align(v: float) -> int:
    return max(ALIGN, int(round(v / ALIGN)) * ALIGN)


def present_size(w: int, h: int, is_original: bool) -> tuple[int, int]:
    if min(w, h) < MIN_SHOWN_SHORT:  # 原图与裁图统一：短边不足 448 放大（≤4x）
        s = min(MIN_SHOWN_SHORT / min(w, h), MAX_UPSCALE)
        w, h = w * s, h * s
    if w * h > AREA_CAP:
        s = (AREA_CAP / (w * h)) ** 0.5
        w, h = w * s, h * s
    return _align(w), _align(h)


def present(img: Image.Image, is_original: bool) -> tuple[Image.Image, tuple[int, int]]:
    tw, th = present_size(img.width, img.height, is_original)
    if (tw, th) != (img.width, img.height):
        img = img.resize((tw, th), Image.LANCZOS)
    return img, (tw, th)


def expand_min_crop(box, W: int, H: int, min_px: int = MIN_CROP_PX):
    x0, y0, x1, y1 = float(box[0]), float(box[1]), float(box[2]), float(box[3])
    expanded = False
    if x1 - x0 < min_px:
        cx = (x0 + x1) / 2
        x0, x1 = cx - min_px / 2, cx + min_px / 2
        expanded = True
    if y1 - y0 < min_px:
        cy = (y0 + y1) / 2
        y0, y1 = cy - min_px / 2, cy + min_px / 2
        expanded = True
    # 先贴边收缩，再在贴边方向向内平移保住 min 边
    x0, y0 = max(0.0, x0), max(0.0, y0)
    x1, y1 = min(float(W), x1), min(float(H), y1)
    if x1 - x0 < min_px:
        if x0 <= 0:
            x1 = min(min_px, float(W))
        else:
            x0 = max(0.0, x1 - min_px)
    if y1 - y0 < min_px:
        if y0 <= 0:
            y1 = min(min_px, float(H))
        else:
            y0 = max(0.0, y1 - min_px)
    return (x0, y0, x1, y1), expanded


def resolve_source(source: str, n_boxes: int) -> int:
    if source == "original_image":
        return 0
    m = re.fullmatch(r"observation_(\d+)", source or "")
    if m and 1 <= int(m.group(1)) < n_boxes:
        return int(m.group(1))
    return -1


class CropEnv:
    """单条题目的裁图环境状态。full 为全分辨率原图，所有裁剪均从原图重切。"""

    def __init__(self, image_path: str):
        self.full = Image.open(image_path).convert("RGB")
        self.W, self.H = self.full.size
        self.boxes = [(0.0, 0.0, float(self.W), float(self.H))]
        img0, shown0 = present(self.full, is_original=True)
        self.images = [img0]           # 按序给模型的呈现图（PIL）
        self.available = ["original_image"]
        self.crops = 0

    def step(self, parsed: dict) -> dict | None:
        """parsed = proto.parse_step 的输出且 kind == 'grounding'。
        返回 {'head','tail','image'} 或 None（非法 source）。"""
        k = resolve_source(parsed.get("source", ""), len(self.boxes))
        if k < 0 or self.crops >= MAX_CROPS:
            return None
        b = parsed["bbox"]  # 相对 source view 的 [0,1] 归一化框
        x0, y0, x1, y1 = self.boxes[k]
        req = (x0 + b[0] * (x1 - x0), y0 + b[1] * (y1 - y0), x0 + b[2] * (x1 - x0), y0 + b[3] * (y1 - y0))
        ab, _ = expand_min_crop(req, self.W, self.H)
        ab = tuple(round(v) for v in ab)
        crop = self.full.crop((int(ab[0]), int(ab[1]), int(ab[2]), int(ab[3])))
        self.crops += 1
        obs_id = f"observation_{len(self.boxes)}"
        img, shown = present(crop, is_original=False)
        self.boxes.append(ab)
        self.images.append(img)
        self.available.append(obs_id)
        fov = [ab[0] / self.W, ab[1] / self.H, ab[2] / self.W, ab[3] / self.H]
        head, tail = proto.receipt_text(
            prev=parsed.get("step", 1), k=len(self.boxes) - 1, source=parsed.get("source", "original_image"),
            fov=fov, src_wh=crop.size, shown_wh=shown, available=list(self.available),
            crops_used=self.crops, budget=MAX_CROPS)
        return {"head": head, "tail": tail, "image": img, "obs_id": obs_id}

    @property
    def remaining(self) -> int:
        return MAX_CROPS - self.crops
