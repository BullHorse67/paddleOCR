import os
import cv2
import numpy as np
import time

from OCRModelFactory import OCRModelFactory, EngineType

# 关键：强制禁用新版 PIR 模式，回归稳定版
os.environ['FLAGS_enable_pir_api'] = '0'
os.environ['FLAGS_enable_new_ir_api'] = '0'

from paddleocr import PaddleOCR, TableRecognitionPipelineV2, FormulaRecognitionPipeline
from paddleocr import PPStructureV3
from paddleocr import LayoutDetection
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from PIL import Image

# from paddleocr.ppstructure.predict_system import save_structure_res
# def _table_engine(img_path):
#     img = cv2.imread(img_path)
#     table_engine = PPStructureV3(
#         show_log=True,
#         image_orientation=True,  # 自动校正方向
#         layout=True,  # 检测表格在页面中的位置
#         table=True,  # 识别表格内部结构
#         lang='ch'  # 中文模型
#     )
#     save_folder = r"C:\Users\HUAWEI\Desktop\dyysai\table"
#     # 3. 执行推理
#     result = table_engine(img)
#     save_structure_res(result, save_folder, os.path.basename(img_path).split('.')[0])
#     # type：text/title/table/table_caption/footer
#     # bbox 坐标
#     # img 三通道 ndarray
#     # res 识别结果 ： text confidence text_region
#     # 5. 打印结果数据结构
#     for region in result:
#         # if region['type'] == 'table':
#         #     # 打印识别出的 HTML 代码（可以直接用 Excel 打开）
#         #     print("--- 检测到表格 ---")
#         #     print(region['res']['html'])
#         print(region["type"])
#         print(region["res"])

def keep_horizontal_bands_and_whiten_rest(img_path, coords_list, save_path):
    """
    遍历坐标列表，保留每个 [x1, y1, x2, y2] 中 y1 到 y2 整个横向范围的图像，
    其余部分全部涂白。

    Args:
        img_path (str): 原图路径
        coords_list (list): 坐标轴列表，格式如 [[x1, y1, x2, y2], [x1, y1, x2, y2], ...]
        save_path (str): 保存结果路径
    """
    # 1. 读取原图
    img = cv2.imread(img_path)
    if img is None:
        print(f"Error: 无法读取图片 {img_path}")
        return

    h, w = img.shape[:2]

    # 2. 创建纯白画布 (尺寸、类型与原图完全一致)
    result = np.full_like(img, 255)

    # 3. 遍历坐标列表进行贴图
    for coords in coords_list:
        # coords 可能是 [x1, y1, x2, y2]
        # 强制转换为 int 解决 TypeError
        try:
            _, y1, _, y2 = coords

            # 使用 int() 确保索引为整数，并进行边界钳制
            y_start = max(0, min(int(y1), int(y2)))
            y_end = min(h, max(int(y1), int(y2)))

            if y_start >= y_end:
                continue

            # 4. 执行贴图
            result[y_start:y_end, :] = img[y_start:y_end, :]
        except Exception as e:
            print(f"处理坐标 {coords} 时出错: {e}")
            continue

    # 5. 保存结果
    cv2.imwrite(save_path, result)
    print(f"处理完成！已保留 {len(coords_list)} 个纵向区段，结果保存至: {save_path}")

def visualize_with_plt(image_path, regions, save_path="layout_debug.png"):
    """
    使用 plt 绘制 LayoutDetection 识别出的文字和表格方框
    :param image_path: 原图路径
    :param regions: 格式如 {"text": [[x1,y1,x2,y2],...], "table": [...]} 的坐标字典
    """
    # 1. 加载图像
    img = Image.open(image_path)
    fig, ax = plt.subplots(1, figsize=(12, 16))
    ax.imshow(img)

    # 2. 定义颜色映射
    # 文本用绿色，表格用红色，标题用蓝色
    color_map = {
        "text": "limegreen",
        "table": "red",
        "title": "blue",
        "image": "yellow",
        "paragraph_title": "brown"
    }

    # 3. 遍历区域并绘制
    for label, coords_list in regions.items():
        color = color_map.get(label, "cyan")  # 默认颜色

        for coords in coords_list:
            # 解析坐标: [x1, y1, x2, y2]
            x1, y1, x2, y2 = coords

            # 计算 matplotlib Rectangle 需用的参数: (x, y), width, height
            width = x2 - x1
            height = y2 - y1

            # 创建矩形框
            # linewidth 线宽, edgecolor 边框颜色, facecolor 填充(设为none)
            rect = patches.Rectangle(
                (x1, y1), width, height,
                linewidth=2,
                edgecolor=color,
                facecolor='none'
            )

            # 将矩形添加到图表
            ax.add_patch(rect)

            # 添加标签文本
            plt.text(
                x1, y1 - 5, label,
                color=color,
                fontsize=10,
                fontweight='bold',
                bbox=dict(facecolor='white', alpha=0.5, edgecolor='none', pad=1)
            )

    # 4. 界面优化
    plt.title(f"Layout Detection Result: {image_path}")
    plt.axis('off')  # 隐藏坐标轴

    # 5. 输出
    plt.tight_layout()
    # plt.savefig(save_path, dpi=200)
    plt.show()


def get_layout_regions(img_path):
    # img = cv2.imread(img_path)
    # 初始化轻量化版面分析模型 (建议使用 PP-DocLayout-S 或 M)
    # model_name 支持: PP-DocLayout-S (极速), PP-DocLayout-M (平衡)
    layout_model = LayoutDetection(model_name="PP-DocLayout-M")

    # 执行识别，layout_nms=True 开启非极大值抑制以消除重叠框
    # DetResult : boxes : cls_id/label/score/coordinate
    #
    results = layout_model.predict(img_path, layout_nms=True)

    regions = {}
    for res in results:
        try:
            save_path = r"C:\Users\HUAWEI\Desktop\dyysai"
            res.save_to_img(save_path=save_path)
            res.save_to_json(save_path=save_path)
            boxes = res['boxes']
        except (TypeError, KeyError):
            # 方案 B: 某些版本中，boxes 实际上就在 res 这一层（res 本身就是那个列表）
            # 这种情况通常出现在你直接打印 res 就能看到一串列表时
            boxes = res

        for box in boxes:
            # 同样注意这里，如果 box 也是字典，请使用 box['coordinate'] 而不是 box.coordinate
            coords = box['coordinate']
            label = box['label']
            score = box['score']
            print(f"检测到: {label}, 坐标: {coords}")

            # 过滤低置信度结果，并将区域归类
            if score > 0.45:
                # 如果 label 不在字典里，初始化为空列表并添加坐标
                regions.setdefault(label, []).append(coords)
                # if label in ['text', 'title']:
                #     regions["text"].append(coords)
                # elif label == 'table':
                #     regions["table"].append(coords)
    print(regions)
    return regions

def tablePPV3(img):
    # pipeline = PPStructureV3()
    # pipeline = PPStructureV3(
    #     device="cpu",
    #     formula_recognition_model_name = "PP-FormulaNet_plus-L",
    #     use_table_recognition=True,  # 默认已启用
    #     use_formula_recognition=False,  # 可按需关闭以提速
    #     use_doc_orientation_classify=False,
    #     use_chart_recognition=False,
    #     use_seal_recognition=False
    # )
    # pipeline = PPStructureV3(
    #     device="cpu",
    #     formula_recognition_model_name=None,
    #     # table_classification_model_name=
    #     use_table_recognition=True,
    #     use_formula_recognition=False,
    #     use_doc_orientation_classify=False,
    #     use_chart_recognition=False,
    #     use_seal_recognition=False
    # )

    """
    use_doc_orientation_classify ：扫描仪扫描时，文件可能会放反（旋转 90° 或 180°）。
    开启此项后，模型会自动判断页面的正反方向并在 OCR 前将其扶正。这比手动旋转图片高效得多。
    use_doc_unwarping ：如果你处理的文件不是平整的扫描件，而是用手机拍摄的弯曲的书页或有折痕的纸张，
    这个模块会通过几何变换将弯曲的文字行“拉直”。
    use_textline_orientation ：与整页分类不同，它针对单行文字。比如文档侧边印有一行垂直的备注。
    开启后，它能识别出这行字是竖着的，从而按正确的顺序读取文字，而不是把每个字拆开识别。
    """
    pipeline = PPStructureV3(
        device="cpu",
        text_detection_model_name="PP-OCRv5_mobile_det",
        text_recognition_model_name="PP-OCRv5_mobile_rec",
        use_doc_orientation_classify=False,
        use_doc_unwarping=False,
        use_textline_orientation=False,
        use_table_recognition=True,
        use_formula_recognition=False,
        use_chart_recognition=False,
        use_seal_recognition=False,
        use_region_detection=False
    )
    # pipeline =TableRecognitionPipelineV2(
    #     text_detection_model_name="PP-OCRv5_mobile_det",
    #     text_recognition_model_name="PP-OCRv5_mobile_rec",
    #     use_doc_orientation_classify=False,
    #     use_doc_unwarping=False,
    #     use_layout_detection=False,
    #     use_ocr_model=True)

    # pipeline = FormulaRecognitionPipeline()
    # output = pipeline.predict(img, use_layout_detection=True)
    output = pipeline.predict(img)
    save_path = r"C:\Users\HUAWEI\Desktop\dyysai\table"
    for res in output:
        # res.print()  ## 打印预测的结构化输出
        # print(res.parsing_res_list)
        # prl = res['parsing_res_list']
        # print(prl)
        # print(type(prl))
        # for p in prl:
        #     print(p)
        #     print(type(p))
        #     print(p.bbox)
            # print(p.get('bbox'))
            # print(p.get('content'))
        res.save_to_json(save_path=save_path)  ## 保存当前图像的结构化json结果
        res.save_to_img(save_path=save_path)
        # res.save_to_markdown(save_path=save_path)  ## 保存当前图像的markdown格式的结果


def paddleOCR(img_path):
    # ocr = PaddleOCR(
    #     # ocr_version = "PP-OCRv5",
    #     text_detection_model_name="PP-OCRv5_mobile_det",
    #     text_recognition_model_name="PP-OCRv5_mobile_rec",
    #     # text_det_box_thresh = 0.4,
    #     use_doc_orientation_classify=True,
    #     use_doc_unwarping=False,
    #     use_textline_orientation=False,
    #     lang="ch"
    # )
    detected_types = [EngineType.TEXT]
    ocr = OCRModelFactory.get_engine(detected_types)
    # ocr = PPStructureV3(
    #             device="cpu",
    #     text_detection_model_name="PP-OCRv5_mobile_det",
    #     text_recognition_model_name="PP-OCRv5_mobile_rec",
    #     use_doc_orientation_classify=False,
    #     use_doc_unwarping=False,
    #     use_textline_orientation=False,
    #             use_table_recognition=False,
    #             use_formula_recognition=False,
    #             use_chart_recognition=False,
    #             use_seal_recognition=False,
    #             use_region_detection=False
    #         )

    """
    use_doc_orientation_classify：文档方向分类可自动识别文档的四个方向（0°、90°、180°、270°），确保文档以正确的方向进行后续处理
    use_doc_unwarping：文本图像矫正的主要目的是针对图像进行几何变换，以纠正图像中的文档扭曲、倾斜、透视变形等问题，以供后续的文本识别进行更加准确
    use_textline_orientation：文本行方向分类模块主要是将文本行的方向区分出来，并使用后处理将其矫正。
    """

    save_path = r"C:\Users\HUAWEI\Desktop\dyysai\table\15"
    result = ocr.predict(img_path)
    for res in result:
        res.print()
        res.save_to_img(save_path)
        res.save_to_json(save_path)
    # 输出结果
    # for idx in range(len(result)):
    #     res = result[idx]
    #     for line in res:
    #         print(f"文本: {line[1][0]}  置信度: {line[1][1]} {line[0]}")


# img_path = r"C:\Users\HUAWEI\Desktop\dyysai\600DPI\GB51099-2015_15.jpg"
# img_path = r"C:\Users\HUAWEI\Desktop\dyysai\600DPI\GB51099-2015_13.jpg"
# img_path = r"C:\Users\HUAWEI\Desktop\dyysai\600DPI\GB51099-2015_109.jpg"
img_path = r"C:\Users\HUAWEI\Desktop\dyysai\600DPI\GB51099-2015_94.jpg"
# img_path = r"C:\Users\HUAWEI\Desktop\dyysai\lightModel\table_regions.png"
# img_path = r"C:\Users\HUAWEI\Desktop\dyysai\table\white_preprocessed_img.jpg"
# img_path = r"C:\Users\HUAWEI\Desktop\dyysai\white.jpg"
# img_path = r"C:\Users\HUAWEI\Desktop\dyysai\600DPI\GB51099-2015_28.jpg"
# original_img = imgBinar.imread_chinese(img_path)
# binary_img = imgBinar.img_preprocess(original_img)

start_time = time.time()

# paddleOCR(img_path)
tablePPV3(img_path)
# get_layout_regions(img_path)
# visualize_with_plt(img_path, regions=get_layout_regions(img_path))
# regions = get_layout_regions(img_path)['formula']
# keep_horizontal_bands_and_whiten_rest(img_path, regions, save_path = r"C:\Users\HUAWEI\Desktop\dyysai\table\white_preprocessed_img.jpg")

end_time = time.time()
print("Execution time:", end_time - start_time, "seconds")