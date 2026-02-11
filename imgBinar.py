import cv2
import numpy as np
import os


def imread_chinese(file_path):
    # 1. 使用 numpy 的 fromfile 从磁盘读取原始字节数据
    # 这就像是在 Java 中读取 FileInputStream 到 byte[]
    img_array = np.fromfile(file_path, dtype=np.uint8)

    # 2. 使用 cv2.imdecode 对内存中的字节数据进行解码
    # cv2.IMREAD_COLOR 表示读取彩色图
    img = cv2.imdecode(img_array, cv2.IMREAD_COLOR)
    return img

def img_preprocess(original_img):
    # --- 核心步骤开始 ---

    # 2. 转换为灰度图 (Grayscale)
    # 二值化的前提是把三通道彩色图(RGB)变成单通道灰度图(亮度图)
    gray_img = cv2.cvtColor(original_img, cv2.COLOR_BGR2GRAY)

    # 3. 应用自适应二值化 (Adaptive Thresholding) - 处理文档的神器
    # 参数详解：
    # - gray_img: 输入的灰度图
    # - 255: 超过阈值的像素设置成的最大值（纯白）
    # - cv2.ADAPTIVE_THRESH_GAUSSIAN_C: 计算局部阈值的方法（高斯加权平均，效果更柔和）
    # - cv2.THRESH_BINARY: 基本的二值化类型（黑白分明）
    # - 25: 【关键参数】邻域块大小（Block Size）。必须是奇数。
    #        它决定了计算阈值时参考周围多大的区域。字越大，这个值要越大。试着在 15, 25, 35 之间调整。
    # - 10: 【关键参数】常数 C。从计算出的平均值中减去这个数。
    #        这个值越大，处理后的图像越“干净”（白色区域更多），越容易把浅色水印过滤掉。试着在 5 到 15 之间调整。
    # binary_img = cv2.adaptiveThreshold(
    #     gray_img,
    #     255,
    #     cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
    #     cv2.THRESH_BINARY,
    #     25,  # Block Size (尝试调整)
    #     10   # C (尝试调整)
    # )

    # 假设 img 是我们读入的灰度图
    # cv2.threshold(src, thresh, maxval, type)
    # 我们把阈值设为 160（这个值需要根据你的水印深浅调整）
    # 凡是亮度 > 160 的，全部变成 255（纯白）
    # 凡是亮度 <= 160 的，全部变成 0（纯黑）
    ret, binary_img = cv2.threshold(gray_img, 160, 255, cv2.THRESH_BINARY)

    return binary_img


def imwrite_chinese(save_path, img):
    """
    支持中文路径的图片保存函数
    """
    try:
        # 1. 检查并创建文件夹（类似 Java 的 mkdirs）
        # 如果路径中的文件夹不存在，写入会失败
        directory = os.path.dirname(save_path)
        if directory and not os.path.exists(directory):
            os.makedirs(directory)

        # 2. 获取文件后缀名 (例如 '.jpg' 或 '.png')
        ext = os.path.splitext(save_path)[1]

        # 3. 将图片格式化编码到内存缓冲区
        # res 是布尔值，im_encode 是编码后的字节流
        res, im_encode = cv2.imencode(ext, img)

        if res:
            # 4. 使用 numpy 的 tofile 将字节流写入磁盘
            # 这避开了 OpenCV 自带的写入函数对路径字符的限制
            im_encode.tofile(save_path)
            print(f"图片已成功保存至: {save_path}")
            return True
        else:
            print("编码失败")
            return False
    except Exception as e:
        print(f"保存图片时发生错误: {e}")
        return False

def optimize_for_ocr(img_path, save_path):
    # 1. 读取并转灰度
    img = imread_chinese(img_path)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    # 2. 图像放大 (提高 OCR 对细小笔画的捕捉能力)
    # 建议放大到原图的 1.5 到 2 倍
    gray = cv2.resize(gray, None, fx=2, fy=2, interpolation=cv2.INTER_CUBIC)

    # 3. 使用大津法 (Otsu) 自动寻找阈值
    # cv2.THRESH_OTSU 会自动计算最佳阈值，忽略你填写的 0
    ret, binary = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

    # 4. 形态学操作：膨胀 (Dilation) -> 相当于“加粗”
    # 创建一个 2x2 的全 1 矩阵作为“笔刷”
    kernel = np.ones((2, 2), np.uint8)
    # 迭代 1 次表示向外扩充一圈。如果文字还是太细，可以改 iterations=2
    # 注意：对于二值化图，文字是黑色(0)，背景是白色(255)
    # OpenCV 的 dilate 默认是扩张亮区。所以我们需要对图像取反，或者使用 erode 黑色区域
    optimized = cv2.erode(binary, kernel, iterations=1)

    # 5. 保存结果
    imwrite_chinese(save_path, optimized)
    print(f"优化完成，自动阈值为: {ret}")
    return optimized


# --- 2. 核心算法逻辑：背景除法去水印 ---
def remove_watermark_norm(img_path, save_path):
    # A. 读取图像
    img = imread_chinese(img_path)
    if img is None: return

    # B. 转灰度
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    cv2.imshow('gray', gray)

    # ---------------------------------------------------------
    # C. 估算背景 (关键步骤)
    # 原理：使用膨胀操作 (Dilate)，让图像中“亮”的区域不断扩张，
    # 只要核 (Kernel) 比文字笔画大，文字就会被周围的白色背景吞噬。
    # 最终得到的 bg_img 就是一张没有文字、只有背景和水印的图。
    # kernel 等于一个橡皮檫抹除掉图像中的细小文字，这个橡皮檫需要比文字的笔画粗才能确保完整抹除
    #
    # ---------------------------------------------------------

    # 调参建议：
    # 如果文字也是大号字（标题），需要把 (25, 25) 调大到 (40, 40)
    # 如果是密集小字，(15, 15) 到 (25, 25) 效果最好
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (80, 80))
    # cv2.imshow('kernel', kernel)

    bg_img = cv2.dilate(gray, kernel)
    # cv2.imshow('bg_img', bg_img)

    # D. 背景归一化 (核心数学公式)
    # 结果 = (原图 / 背景) * 255
    # - 背景区域：200/200 = 1 -> 255 (变白)
    # - 水印区域：180/180 = 1 -> 255 (变白，水印消失！)
    # - 文字区域：50/200 = 0.25 -> 63 (依然是深色，文字保留！)
    norm_img = cv2.divide(gray, bg_img, scale=255)
    # cv2.imshow('norm_img', norm_img)

    # E. 最终二值化
    # 经过上面一步，背景已经非常干净了（纯白），这时候用 Otsu 自动阈值非常精准
    ret, binary = cv2.threshold(norm_img, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

    # (可选) F. 最后微调：稍微腐蚀一点点，让字体更饱满
    # 这一步视情况而定，如果字太细可以开启
    # kernel_refine = np.ones((2, 2), np.uint8)
    # binary = cv2.erode(binary, kernel_refine, iterations=1)

    # 保存结果
    success = imwrite_chinese(save_path, binary)
    if success:
        print(f"[处理完成]：\n -> 原图：{img_path}\n -> 结果：{save_path}")
        print(f" -> 自动计算的阈值：{ret} (背景已被归一化处理)")

    return binary

# --- 核心改进函数 ---
def preserve_text_remove_watermark(img_path, save_path):
    img = imread_chinese(img_path)
    if img is None: return
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    # 1. 更加精细的背景估计：使用中值滤波
    # 这个核的大小决定了“抹除”文字的程度。尝试 11, 21, 31 (必须是奇数)
    # 中值滤波能比膨胀更好地处理水印边缘，减少后期黑点的产生
    bg_estimated = cv2.medianBlur(gray, 21)

    # 2. 归一化处理（除法）
    # 这一步将背景强制拉平，水印会被大幅度削弱
    norm_img = cv2.divide(gray, bg_estimated, scale=255)

    # 3. 局部自适应二值化 (针对丢失笔画的核心改进)
    # 我们不使用全局阈值，而是让算法在 31x31 的小范围内找文字
    # 参数 C (最后一个参数 15) 是关键：
    # - 调大 C：会过滤更多浅色噪点（解决水印黑点问题）
    # - 调小 C：会保留更多细弱笔画（解决正文丢失问题）
    binary = cv2.adaptiveThreshold(
        norm_img,
        255,
        cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
        cv2.THRESH_BINARY,
        31, # Block Size: 考察区域大小
        1000  # C: 阈值补偿常数
    )

    # 保存结果
    imwrite_chinese(save_path, binary)
    print(f"处理完成，保存至: {save_path}")

    return binary

# stack overflow 版本
def wartermark_remove(filename):
    # img = imread_chinese(img_path)
    img = cv2.imread(filename)

    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    for i in range(7):
        kernel2 = cv2.getStructuringElement(cv2.MORPH_ELLIPSE,
                                            (2 * i + 1, 2 * i + 1))
        bg = cv2.morphologyEx(gray, cv2.MORPH_CLOSE, kernel2)
        bg = cv2.morphologyEx(bg, cv2.MORPH_OPEN, kernel2)

        dif = cv2.subtract(bg, gray)

        bw = cv2.threshold(dif, 0, 255, cv2.THRESH_BINARY_INV | cv2.THRESH_OTSU)[1]
        dark = cv2.threshold(bg, 0, 255, cv2.THRESH_BINARY_INV | cv2.THRESH_OTSU)[1]

        darkpix = gray[np.where(dark > 0)]

        darkpix = cv2.threshold(darkpix, 0, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU)[1]

        bw[np.where(dark > 0)] = darkpix.T

        cv2.imshow('Adaptive Binary (Cleaned)', bw)

        print("请查看弹出的窗口对比效果，按任意键退出...")
        cv2.waitKey(0)
        cv2.destroyAllWindows()

    # 保存结果
    # imwrite_chinese(save_path, bw)
    return bw

def watermark_remove2(img_path):
    im = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)

    # 1. estimate watermark
    r = 4
    k = r * 2 + 1
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (k, k))
    watermark = cv2.morphologyEx(src=im, op=cv2.MORPH_CLOSE, kernel=kernel)

    # 2. calculate mask
    (th, mask) = cv2.threshold(watermark, 0, 255, cv2.THRESH_BINARY_INV | cv2.THRESH_OTSU)
    mask = cv2.dilate(mask, kernel=None, iterations=1)
    mask = mask.astype(bool)

    # 3. estimate watermark's color
    watermark_value = np.median(im[mask], axis=0)

    # 4. correct watermarked pixels
    result = im.copy()
    correction_factor = np.float32(255 / watermark_value)
    result[mask] = (result[mask] * correction_factor).clip(0, 255).astype('u1')
    return result


def clean_watermark_with_refinement(img_path):
    # 1. 读取原图
    img = cv2.imread(img_path)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    # ================= 步骤一：获取高精度的 Mask (你的去水印逻辑) =================
    # 我们不需要循环，直接选一个经验上最好的核大小，比如 i=2 或 i=3 时的效果
    # 这里的 Kernel 大小决定了你是要把多粗的笔画保留下来
    kernel_size = 15
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (kernel_size, kernel_size))

    # 形态学获取背景
    bg = cv2.morphologyEx(gray, cv2.MORPH_CLOSE, kernel)
    bg = cv2.morphologyEx(bg, cv2.MORPH_OPEN, kernel)

    # 差分
    dif = cv2.subtract(bg, gray)

    # 获取初步 Mask (二值图)
    # 使用自适应阈值，能更好地保留文字骨架
    mask = cv2.adaptiveThreshold(dif, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                                 cv2.THRESH_BINARY, 25, 5)

    # ================= 步骤二：Mask 优化 (关键步骤) =================
    # 原始的 Mask 可能有噪点（没去干净的水印），或者字太细
    # 1. 膨胀一点点：让 Mask 稍微比字大一圈，保证把字边缘的模糊像素也包进去
    dilate_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
    mask = cv2.dilate(mask, dilate_kernel, iterations=1)

    # ================= 步骤三：利用原图回填 (你的 Idea) =================
    # 创建一个纯白背景
    result = np.full_like(img, 255)

    # 核心逻辑：dst(I) = src(I) if mask(I) != 0 else 255
    # copyTo 是 C++ 写法，Python 中我们用 bitwise 操作

    # 1. 从原图中把字“抠”出来 (背景变黑，字保留原色)
    # bitwise_and 需要 mask 是单通道
    text_part = cv2.bitwise_and(img, img, mask=mask)
    cv2.imshow('text_part', text_part)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

    # 2. 把抠出来的部分的背景变成白色
    # mask_inv 是 mask 的反色：字是黑(0)，背景是白(255)
    mask_inv = cv2.bitwise_not(mask)
    cv2.imshow('mask_inv', mask_inv)
    cv2.waitKey(0)
    cv2.destroyAllWindows()


    # 在白背景上把字的位置挖空
    white_bg_with_hole = cv2.bitwise_and(result, result, mask=mask_inv)

    # 3. 拼合：(原图的字) + (挖空了字的白纸)
    final_img = cv2.add(text_part, white_bg_with_hole)

    return final_img


def watermark_remove_multiscale(img_path):
    # 1. 读取图像 (复用你的逻辑)
    # img = imread_chinese(img_path)
    img = cv2.imread(img_path)
    if img is None:
        return None

    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    # 2. 初始化最终结果容器
    # 创建一个和原图一样大小的全白图像 (所有像素 255)
    # 这里的逻辑是：我们通过“做减法”把黑色的字一点点印上去
    final_result = np.full(gray.shape, 255, dtype=np.uint8)

    # 3. 循环叠加 (多尺度处理)
    # 你的原代码是 range(5)，即核大小为 1, 3, 5, 7, 9
    for i in range(5):
        # === 以下为你提供的原始核心处理逻辑 ===
        kernel_size = 2 * i + 1
        kernel2 = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (kernel_size, kernel_size))

        # 形态学操作估算背景
        bg = cv2.morphologyEx(gray, cv2.MORPH_CLOSE, kernel2)
        bg = cv2.morphologyEx(bg, cv2.MORPH_OPEN, kernel2)

        # 差分：背景 - 原图
        dif = cv2.subtract(bg, gray)

        # 第一次二值化 (提取主体)
        bw = cv2.threshold(dif, 0, 255, cv2.THRESH_BINARY_INV | cv2.THRESH_OTSU)[1]

        # 暗部细节召回 (Original logic)
        dark = cv2.threshold(bg, 0, 255, cv2.THRESH_BINARY_INV | cv2.THRESH_OTSU)[1]
        darkpix = gray[np.where(dark > 0)]
        darkpix = cv2.threshold(darkpix, 0, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU)[1]

        # 将召回的细节填入 bw
        # 注意：这里 bw 可能包含部分灰度值，不完全是 0/255
        bw[np.where(dark > 0)] = darkpix.T

        # === 核心修改点：叠加逻辑 ===

        # 利用 np.minimum 进行像素级比较
        # 逻辑：对于每个像素，取 final_result 和 当前 bw 中较黑的那个(值较小的)
        # 如果某一次循环提取到了字(黑色0)，那么最终结果该位置就会永久变成黑色
        final_result = np.minimum(final_result, bw)

        # 调试代码 (可选)：看每一次叠加后的效果
        cv2.imshow(f'Step_{i}', final_result)
        cv2.waitKey(0)
        cv2.destroyAllWindows()

    return final_result


def remove_watermark_with_gaussian(img_path):
    # 1. 读取图像
    # img = imread_chinese(img_path)
    img = cv2.imread(img_path)
    if img is None:
        return None

    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    # 2. 高斯模糊 (关键步骤)
    # ksize (核大小): 必须大于文字的笔画和字号，通常取奇数。
    # 如果你的字很大，这个值要设得更大 (比如 51, 51)
    # 如果字很小，可以用 (15, 15)
    # 核心思路：把字“糊”没了，只剩下背景
    ksize = (45, 45)
    blur = cv2.GaussianBlur(gray, ksize, 0)

    # [调试用] 你可以把 blur 保存下来看看，应该是一张看不清字，但能看清水印轮廓的图
    # cv2.imwrite('debug_blur.jpg', blur)

    # 3. 除法运算 (核心算法)
    # 算法公式：Result = (Gray / Blur) * 255
    # cv2.divide 执行的是饱和运算，会自动处理 0-255 的范围
    # scale=255 是将结果归一化到 0-255 亮度区间
    normalized = cv2.divide(gray, blur, scale=255)

    # 4. 增强对比度 (可选)
    # 经过除法后，字可能会变淡（比如从纯黑变成深灰），需要拉黑
    # 使用 Otsu 阈值自动二值化，或者自适应阈值

    # 方案 A: 直接 Otsu 二值化 (最狠，直接黑白)
    ret, final_result = cv2.threshold(normalized, 0, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU)

    # 方案 B: 伽马校正 (温和，保留边缘抗锯齿，适合后续 OCR)
    # 伽马 < 1 提亮，伽马 > 1 变暗。这里我们要让字更黑，背景保持白。
    # 但由于我们已经把背景变成了纯白(255)，我们其实只需要简单的阈值切分
    # final_result = cv2.adaptiveThreshold(normalized, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
    #                                      cv2.THRESH_BINARY, 25, 10)

    return final_result


def remove_watermark_enhance_text(img_path):
    img = cv2.imread(img_path)
    if img is None:
        return None
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    # 1. 伽马变换 (保留之前的逻辑，去除浅灰水印)
    # 稍微降低一点 gamma 值，防止对文字边缘伤害太大
    # 之前可能设了 4.0，这次试一下 2.5 - 3.0
    gamma = 5.0
    invGamma = 1.0 / gamma
    table = np.array([((i / 255.0) ** invGamma) * 255 for i in np.arange(0, 256)]).astype("uint8")
    bleached = cv2.LUT(gray, table)

    # ================= 新增步骤：USM 锐化 (Unsharp Masking) =================
    # 原理：原图 + (原图 - 模糊图) * 系数
    # 这能显著增强文字边缘的锐度，对抗模糊
    gaussian = cv2.GaussianBlur(bleached, (0, 0), 3.0)
    # addWeighted 公式: src1 * alpha + src2 * beta + gamma
    # 这里给予原图 1.5 倍权重，减去 0.5 倍的模糊图，实现锐化
    sharpened = cv2.addWeighted(bleached, 1.3, gaussian, -0.5, 0)

    # ================= 改进步骤：阈值选择 =================
    # 你的痛点：Otsu 切掉了边缘。
    # 方案 1: 稍微放宽 Otsu 的阈值
    otsu_val, _ = cv2.threshold(sharpened, 0, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU)
    # 手动把 Otsu 算出来的阈值提高一点（比如 +15），让更多灰色被判定为黑色
    # 注意：阈值越高(越接近255)，被判定为黑色的像素越多
    # adjusted_thresh_val = min(otsu_val + 20, 255)
    # _, binary = cv2.threshold(sharpened, adjusted_thresh_val, 255, cv2.THRESH_BINARY)

    # 方案 2 (替代方案，通常更好): 自适应阈值
    # 如果上面的 binary 还是虚，请注释掉上面两行，解开下面这一行：
    binary = cv2.adaptiveThreshold(sharpened, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                                   cv2.THRESH_BINARY, 25, 15)

    # ================= 修复步骤：形态学微调 =================
    # 如果字还是有点断断续续，用“腐蚀”操作（注意：在黑底白字中是膨胀，在白底黑字中是腐蚀）
    # OpenCV 的习惯是处理“白色前景”。
    # 我们的图通常是“白底黑字”，所以要让黑色变粗，其实是对白色背景做“腐蚀”

    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (2, 2))
    # Erode 会让黑色区域扩张 (加粗文字)
    final_result = cv2.erode(binary, kernel, iterations=1)

    return final_result


import cv2
import numpy as np


def clean_watermark_pure_strategy(img_path):
    # 1. 读取图像
    img = cv2.imread(img_path)
    if img is None:
        return None
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    # 2. 滤波预处理：使用中值滤波 (Median Blur)
    # 这一步对应你的思考。
    # 优势：中值滤波是非线性的，它能在去噪的同时，保留文字边缘的“台阶跳变”，不会像高斯那样把字弄虚。
    # ksize 必须是奇数。3 或 5 适合大多数文档。
    denoised = cv2.medianBlur(gray, 1)

    # 3. 核心二值化：自适应阈值 (Adaptive Threshold)
    # 摒弃 Otsu，改用局部对比度。
    # blockSize=25: 观察邻域大小。
    # C=10: 阈值偏置。意思是：只有比邻域平均值暗 10 个单位以上的像素，才算字。
    # 调节技巧：
    # - 如果水印残留多 -> 增大 C (比如 15, 20)
    # - 如果字断了 -> 减小 C (比如 5, 8)
    binary = cv2.adaptiveThreshold(denoised, 255,
                                   cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                                   cv2.THRESH_BINARY,
                                   25, 15)

    #
    # 此时图像应该非常清晰，但可能布满细小的“雪花”噪点（被切碎的水印）

    # 4. 物理层过滤：基于连通域面积 (Contours)
    # 这是最后一道防线。我们不修改像素值，而是直接把“看起来不像字”的东西扔掉。

    # 查找所有黑色连通块（注意：OpenCV通常处理白色前景，所以这里我们针对binary是白底黑字的情况，
    # 查找轮廓时通常需要反转，或者直接找，但需要注意层级）

    # 为了方便查找轮廓，先反转成“黑底白字”
    binary_inv = cv2.bitwise_not(binary)
    contours, _ = cv2.findContours(binary_inv, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    # 创建一个纯净的掩膜
    final_mask = np.zeros_like(gray)

    for cnt in contours:
        area = cv2.contourArea(cnt)

        # === 核心逻辑：面积筛选 ===
        # 极小的点 (area < 5): 噪点、水印残留 -> 丢弃
        # 极大的块 (area > 1000): 可能是边框、大Logo -> 丢弃 (根据实际情况调整)
        # 正常的字: 通常在 10 ~ 500 之间 (取决于分辨率)
        if 5 < area < 2000:
            cv2.drawContours(final_mask, [cnt], -1, 255, -1)

    # 此时 final_mask 是黑底白字的纯净图
    # 转回白底黑字适应 OCR
    final_result = cv2.bitwise_not(final_mask)

    return final_result

# --- 核心步骤结束 ---
# 1. 读取原始图片
img_path = r"C:\Users\HUAWEI\Desktop\dyysai\600DPI\GB51099-2015_15.jpg"
save_path = r"C:\Users\HUAWEI\Desktop\dyysai\cleaned_image.jpg"
# 以彩色模式读取，方便后续可能的颜色分析
# original_img = imread_chinese(img_path)
# original_img = cv2.imread(img_path)
# original_img = remove_watermark_norm(img_path, save_path)
# original_img = preserve_text_remove_watermark(img_path, save_path)
# original_img = wartermark_remove(img_path)
# original_img = watermark_remove2(img_path)
# original_img = clean_watermark_with_refinement(img_path)
# original_img = watermark_remove_multiscale(img_path)
# original_img = remove_watermark_with_gaussian(img_path)
# original_img = remove_watermark_enhance_text(img_path)
original_img = clean_watermark_pure_strategy(img_path)


# binary_img = img_preprocess(original_img)
#
# 保存处理后的图片，供 OCR 使用
# imwrite_chinese(save_path, binary_img)
cv2.imwrite(save_path, original_img)
print(f"已保存处理后的图片至：{save_path}，现在可以用它去跑 OCR 了！")

# optimized = optimize_for_ocr(img_path, save_path)

# 4. 显示对比结果 (按任意键关闭窗口)
# 注意：在服务器端无界面环境下不要运行 cv2.imshow
# cv2.imshow('Original', original_img)
# cv2.imshow('Gray', gray_img)
cv2.imshow('Adaptive Binary (Cleaned)', original_img)
#
# print("请查看弹出的窗口对比效果，按任意键退出...")
cv2.waitKey(0)
cv2.destroyAllWindows()

