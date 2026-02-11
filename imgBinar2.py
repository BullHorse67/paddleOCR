import cv2
import numpy as np
import os

import matplotlib.pyplot as plt
from skimage.filters import threshold_sauvola


def mask_area_below_text(image_path, output_path=None):
    """
    自动识别图像中文本块的底部边界，并将下方的区域全部涂白。
    """
    # 1. 读取图像并转换为灰度
    img_original = cv2.imread(image_path)
    if img_original is None:
        raise ValueError(f"无法读取图像: {image_path}")
    gray = cv2.cvtColor(img_original, cv2.COLOR_BGR2GRAY)
    h, w = gray.shape[:2]

    # 2. 二值化处理 (为了突出文字特征)
    # 使用 Otsu's 阈值法自动寻找最佳阈值，反转图像使得文字为白色，背景为黑色
    # 也可以根据实际情况使用 adaptiveThreshold
    _, binary = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)

    # --- 核心步骤：形态学操作 ---

    # 定义结构元素 (Kernel)。
    # 关键点：设置一个宽而扁的核，目的是让同一行的文字横向粘连在一起，
    # 但不要让不同行的文字纵向粘连。
    # 对于一般分辨率的文档图片，(50, 5) 左右是一个经验起始值。
    # 如果你的图片分辨率极高，可能需要增大这个值，例如 (100, 10)。
    kernel_width = w // 20  # 动态调整，约为宽度的1/20
    kernel_height = 5  # 保持较小的高度，避免跨行粘连
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (kernel_width, kernel_height))

    # 进行膨胀操作 (Dilation)
    # 文字笔画会变粗并连接成片
    dilated_img = cv2.dilate(binary, kernel, iterations=2)

    # 3. 查找轮廓
    # 寻找膨胀后白色区域的轮廓
    contours, _ = cv2.findContours(dilated_img, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    # 4. 筛选轮廓并找到最底部的文本边界
    bottom_y_candidate = 0

    for contour in contours:
        # 获取边界框
        x, y, rect_w, rect_h = cv2.boundingRect(contour)

        # --- 启发式筛选规则 (需要根据实际数据调整) ---
        # 规则1: 忽略太小的噪点区域
        if rect_w < w / 10 or rect_h < 10:
            continue

        # 规则2: 忽略纵向跨度太大的区域 (可能是侧边的纵向水印或装订线)
        if rect_h > h / 2:
            continue

        # 规则3 (可选): 忽略横向过于贯通整个页面的细长条 (可能是页眉页脚线)
        # if rect_w > w * 0.95 and rect_h < 20:
        #     continue

        # 计算当前这个文本块的底部 Y 坐标
        current_bottom = y + rect_h

        # 更新全局最靠下的边界
        if 856 > current_bottom > bottom_y_candidate:
            bottom_y_candidate = current_bottom

    # 为了安全起见，可以在计算出的底部再往下留一点点缓冲距离 (Padding)
    padding = 10
    final_cutoff_y = min(bottom_y_candidate + padding, h - 1)

    print(f"检测到文本块底部 Y 坐标: {bottom_y_candidate}, 最终截断 Y 坐标: {final_cutoff_y}")

    # 5. 绘制遮罩
    # 复制原图进行修改
    img_masked = img_original.copy()

    # 如果找到了有效的底部边界（防止一张全白或全黑图导致 bottom_y 为 0）
    if final_cutoff_y > padding:
        # cv2.rectangle 参数: 图像, 左上角(x1,y1), 右下角(x2,y2), 颜色(BGR), 线宽(-1表示填充)
        # 将从截止线到图像最底部的区域填充为纯白色
        cv2.rectangle(img_masked, (0, final_cutoff_y), (w, h), (255, 255, 255), -1)
    else:
        print("警告：未能有效检测到文本块，未进行遮罩处理。")

    # 保存或显示结果
    if output_path:
        cv2.imwrite(output_path, img_masked)
        print(f"处理后的图像已保存至: {output_path}")

    # --- 可视化调试过程 (可选，开发时非常有帮助) ---
    # 这一步可以让你看到形态学操作处理成了什么样子，方便调整 Kernel 大小
    plt.figure(figsize=(10, 5))
    plt.subplot(121), plt.imshow(dilated_img, cmap='gray'), plt.title('Dilated View (Morphology)')
    plt.subplot(122), plt.imshow(cv2.cvtColor(img_masked, cv2.COLOR_BGR2RGB)), plt.title('Final Masked Result')
    plt.show()
    # -------------------------------------------

    return img_masked


def apply_sauvola_median_upscaled(image_path, output_path):
    # 1. 读取原始灰度图
    original_img = cv2.imread(image_path, 0)

    # -------------------------------------------------------------
    # 【关键步骤】：图像放大 (Upscaling)
    # 作用：增加笔画的像素宽度，让中值滤波不敢轻易“吃掉”文字细节
    # fx=2, fy=2 表示长宽各放大 2 倍
    # INTER_CUBIC (三次样条插值) 能保持放大的平滑度，避免锯齿
    # -------------------------------------------------------------
    upscaled_img = cv2.resize(original_img, None, fx=5, fy=5, interpolation=cv2.INTER_CUBIC)

    # 2. 在放大图上做 Sauvola
    # 注意：因为图变大了，Window Size 也要相应变大，否则视野会变窄
    # 原来用 25，现在建议用 51 左右
    thresh = threshold_sauvola(upscaled_img, window_size=11, k=0.03)
    binary = (upscaled_img > thresh) * 255
    binary = binary.astype('uint8')

    # 3. 使用中值滤波
    # 现在可以在高清图上放心地用 3x3 甚至 5x5 的核
    # 因为笔画变粗了，滤波只会去除非文字的孤立噪点
    denoised_upscaled = cv2.medianBlur(binary, 9)

    # 4. (可选) 缩回原尺寸
    # 如果业务需要原尺寸图片，再缩回去；
    # 但对于 OCR 识别，保留大图通常识别率更高！建议不缩回。
    # final_result = cv2.resize(denoised_upscaled, None, fx=0.5, fy=0.5, interpolation=cv2.INTER_NEAREST)
    final_result = denoised_upscaled

    # cv2.imshow('final_result', final_result)
    # cv2.waitKey(0)

    # 保存结果 (请务必使用 png 以防压缩模糊)
    cv2.imwrite(output_path, final_result)
    print(f"处理完成，结果已保存: {output_path}")


def apply_sauvola_threshold(image_path, output_path, window_size=3, k=0.15):
    """
    使用 Sauvola 算法对图像进行二值化处理

    参数:
        image_path (str): 输入图片路径
        output_path (str): 输出图片路径
        window_size (int): 局部窗口大小（必须是奇数），决定了算法"看"多大的范围
        k (float): 敏感度参数，范围通常在 0.2 - 0.5 之间
    """

    # 1. 读取图像
    # 注意：OCR 预处理必须先转为灰度图 (0代表灰度模式读取)
    image = cv2.imread(image_path, cv2.COLOR_BGR2GRAY)

    if image is None:
        print(f"Error: 无法找到或读取图片 {image_path}")
        return

    # denoised = cv2.medianBlur(image, 3)

    # blur = cv2.GaussianBlur(image, (11, 11), 0)

    # 2. 核心：计算 Sauvola 阈值曲面
    # scikit-image 的 threshold_sauvola 会返回一个与原图同尺寸的矩阵
    # 矩阵中每个点代表该像素点对应的阈值
    thresh_sauvola_map = threshold_sauvola(image, window_size=window_size, k=k)

    # 3. 应用阈值进行二值化
    # 逻辑：如果 像素值 > 阈值，则为背景(True/白色)；否则为前景文字(False/黑色)
    # 这一步生成的是一个 Boolean 类型的矩阵
    binary_image = image > thresh_sauvola_map

    # 4. 类型转换 (关键步骤)
    # OCR 引擎和 OpenCV 需要的是 uint8 (0-255) 的数字图像，而不是 Boolean
    # 将 True(1) 转为 255 (白)，False(0) 转为 0 (黑)
    binary_image_uint8 = (binary_image * 255).astype('uint8')

    cv2.imshow('Result', binary_image_uint8)
    cv2.waitKey(0)

    # 5. 保存结果
    # cv2.imwrite(output_path, binary_image_uint8)
    # print(f"处理完成，结果已保存至: {output_path}")

def watermark_remove_strict(img_path):
    # 1. 读取图片并转灰度
    img = cv2.imread(img_path)
    if img is None:
        return None
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    # 2. 中值滤波 (Median Blur)
    # 作用：去除椒盐噪声（如细小的水印噪点），同时保留文字边缘。
    # 参数 3 表示核大小为 3x3。如果噪点较多，可改为 5。必须是奇数。
    # denoised = cv2.medianBlur(gray, 3)

    # 2. 高斯滤波 (Gaussian Blur)
    # 作用：平滑图像，降低随机噪声。
    # (5, 5) 是核大小 (ksize)，必须是奇数。
    # 0 表示标准差由核大小自动计算。
    # 如果觉得模糊过度（字变虚），可以改为 (3, 3)。
    # 如果觉得去噪不够（水印噪点多），可以改为 (7, 7)。
    # blurred = cv2.GaussianBlur(gray, (5, 5), 0)

    # 3. Otsu 二值化 (Otsu's Binarization)
    # 作用：自动计算全局最佳阈值，将图像切分为黑白。
    # 0 是占位符，cv2.THRESH_OTSU 会自动计算出真正的阈值并覆盖它。
    # ret, binary = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU)

    binary = cv2.adaptiveThreshold(gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                                         cv2.THRESH_BINARY, 25, 30)

    return binary

# 使用示例
file_path = r"C:\Users\HUAWEI\Desktop\dyysai\600DPI\GB51099-2015_15.jpg"
# file_path = r"C:\Users\HUAWEI\Desktop\dyysai\600DPI\GB51099-2015_01.jpg"
# result = watermark_remove_strict(file_path)
# cv2.imshow('Result', result)
# cv2.waitKey(0)
save_path = r"C:\Users\HUAWEI\Desktop\dyysai\imgx5_11_0.07.jpg"

# apply_sauvola_threshold(file_path, save_path)
# apply_sauvola_median_upscaled(file_path, save_path)
img = mask_area_below_text(file_path, r"C:\Users\HUAWEI\Desktop\dyysai\white.jpg")

cv2.imshow('Adaptive Binary (Cleaned)', img)
#
# print("请查看弹出的窗口对比效果，按任意键退出...")
cv2.waitKey(0)
cv2.destroyAllWindows()
