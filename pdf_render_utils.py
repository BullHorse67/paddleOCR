import cv2
import fitz
import numpy as np
from matplotlib import pyplot as plt


class PDFRenderUtils:
    """PDF 渲染工具类。"""

    @staticmethod
    def render_page_image(doc, page_idx: int, dpi: int = 200):
        """将 PDF 指定页渲染为 OpenCV BGR 图像。"""
        page = doc.load_page(page_idx)
        scale = dpi / 72
        pix = page.get_pixmap(matrix=fitz.Matrix(scale, scale))
        img = np.frombuffer(pix.samples, dtype=np.uint8).reshape(pix.h, pix.w, pix.n)

        if pix.n == 4:
            return cv2.cvtColor(img, cv2.COLOR_RGBA2BGR)
        return cv2.cvtColor(img, cv2.COLOR_RGB2BGR)

    @staticmethod
    def visualize_cv2_image(image, save_path: str = None, show: bool = False, window_name: str = "image"):
        """可视化 cv2 图像：支持保存到文件，展示时使用 plt 并自动缩放。"""
        if image is None:
            raise ValueError("image 不能为空")

        if save_path:
            cv2.imwrite(save_path, image)

        if show:
            h, w = image.shape[:2]
            max_side = max(h, w)
            scale = 1.0 if max_side <= 1600 else 1600 / max_side
            fig_w = max(4, (w * scale) / 100)
            fig_h = max(4, (h * scale) / 100)

            rgb_image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            plt.figure(figsize=(fig_w, fig_h))
            plt.imshow(rgb_image)
            plt.title(window_name)
            plt.axis('off')
            plt.tight_layout()
            plt.show()

        return image

    @staticmethod
    def _normalize_ratio(ratio: float) -> float:
        """兼容 0~1 与 0~100 两种比例输入，统一转换为 0~1。"""
        if ratio < 0:
            raise ValueError("裁剪比例不能为负数")
        if ratio > 100:
            raise ValueError("裁剪比例不能大于 100%")
        if ratio > 1:
            return ratio / 100.0
        return ratio

    @staticmethod
    def crop_edges_by_aspect_ratio(
        image,
        top_ratio: float = 0,
        bottom_ratio: float = 0,
        left_ratio: float = 0,
        right_ratio: float = 0,
    ):
        """
        按顶部/底部/左侧/右侧比例裁剪图像边缘。
        - 裁剪顶部长度：H * top_ratio
        - 裁剪底部长度：H * bottom_ratio
        - 裁剪左侧长度：W * left_ratio
        - 裁剪右侧长度：W * right_ratio
        比例为 0 时表示该边不裁剪。
        """
        if image is None:
            raise ValueError("image 不能为空")

        h, w = image.shape[:2]
        if h == 0 or w == 0:
            raise ValueError("输入图像尺寸非法")

        top_ratio = PDFRenderUtils._normalize_ratio(top_ratio)
        bottom_ratio = PDFRenderUtils._normalize_ratio(bottom_ratio)
        left_ratio = PDFRenderUtils._normalize_ratio(left_ratio)
        right_ratio = PDFRenderUtils._normalize_ratio(right_ratio)

        top = int(h * top_ratio)
        bottom = int(h * bottom_ratio)
        left = int(w * left_ratio)
        right = int(w * right_ratio)

        y1, y2 = top, h - bottom
        x1, x2 = left, w - right

        if y1 >= y2 or x1 >= x2:
            raise ValueError("裁剪比例过大，导致图像尺寸无效")

        return image[y1:y2, x1:x2]

    @staticmethod
    def crop_edges_by_ratio(image, ratio: float = 0):
        """按统一比例同时裁剪上下左右四条边。"""
        normalized_ratio = PDFRenderUtils._normalize_ratio(ratio)
        return PDFRenderUtils.crop_edges_by_aspect_ratio(
            image,
            top_ratio=normalized_ratio,
            bottom_ratio=normalized_ratio,
            left_ratio=normalized_ratio,
            right_ratio=normalized_ratio,
        )
