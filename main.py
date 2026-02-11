# This is a sample Python script.
import cv2


# Press Shift+F10 to execute it or replace it with your code.
# Press Double Shift to search everywhere for classes, files, tool windows, actions, and settings.


# --- 核心功能函数 ---
def whiten_horizontal_region(img_path: str, save_path: str, y1: int, y2: int):
    """
    将图片指定纵向范围 [y1, y2) 的区域横向涂白。

    Args:
        img_path: 原图路径
        save_path: 保存路径
        y1: 起始 Y 坐标 (包含)
        y2: 结束 Y 坐标 (不包含)
    Returns:
        bool: 处理是否成功
    """
    # 1. 读取图片
    img = cv2.imread(img_path)
    if img is None:
        return False

    # 2. 获取图像高度 (Shape 的第一个元素是高度)
    h = img.shape[0]
    print(f"图像高度: {h}, 请求涂白区域: y[{y1} -> {y2}]")

    # 3. 坐标安全验证与钳制 (Clamping)
    # 确保 y1 不小于 0
    y1_safe = max(0, y1)
    # 确保 y2 不大于高度，且 y2 至少要大于等于 y1_safe
    y2_safe = min(h, max(y1_safe, y2))

    # 如果区域无效（例如 y1 >= y2，或者 y1 已经超过了图片高度）
    if y1_safe >= y2_safe:
        print("警告：涂白区域无效 (y1 >= y2 或超出图像范围)，未做任何修改。")
        # 依然保存原图，或者返回 False，视业务需求而定。这里选择保存。
        # cv2.imwrite(save_path, img)
        return True

    print(f"实际执行涂白区域 (修正后): y[{y1_safe} -> {y2_safe}]")

    # 4. 核心操作：NumPy 切片赋值涂白
    # img[行切片, 列切片] = 值
    # 这里利用 NumPy 广播机制，赋值 255 会让 BGR 三个通道都变为 255（纯白）
    img[y1_safe:y2_safe, :] = 255

    # 5. 保存结果
    if cv2.imwrite(save_path, img):
        print(f"处理成功，已保存至: {save_path}")
        return True
    else:
        return False


# Press the green button in the gutter to run the script.
# if __name__ == '__main__':


# See PyCharm help at https://www.jetbrains.com/help/pycharm/
