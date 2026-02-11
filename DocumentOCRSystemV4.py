import concurrent
import multiprocessing
import os
import time

import fitz
import cv2
import numpy as np
from typing import List, Dict, Any

from paddleocr import PaddleOCR, PPStructureV3

from OCRLightModelFactory import OCRLightModelFactory
from OCRModelFactory import OCRModelFactory, EngineType

"""
并发版
"""


class DocumentOCRSystemV4:
    def __init__(self, output_dir: str):
        self.output_dir = output_dir
        if not os.path.exists(output_dir):
            os.makedirs(output_dir)

        self.label_map = {
            'text': EngineType.TEXT,
            'number': EngineType.TEXT,
            'paragraph_title': EngineType.TEXT,
            'figure_title': EngineType.TEXT,
            'table_title': EngineType.TEXT,
            'table': EngineType.TABLE,
            'formula': EngineType.FORMULA,
            'formula_number': EngineType.TEXT
        }

    @staticmethod
    def _split_page_chunks(total_pages: int, process_count: int) -> List[List[int]]:
        """按进程数平均切分页索引，保证页码连续且覆盖完整。"""
        if total_pages <= 0:
            return []
        process_count = max(1, min(process_count, total_pages))

        base = total_pages // process_count
        remain = total_pages % process_count

        chunks = []
        start = 0
        for i in range(process_count):
            size = base + (1 if i < remain else 0)
            end = start + size
            chunks.append(list(range(start, end)))
            start = end
        return [chunk for chunk in chunks if chunk]

    def _process_page_chunk(self, pdf_path: str, page_indices: List[int], run_folder: str, extension: str):
        """每个子进程处理一个页块，单页异常跳过。"""
        infos = []
        doc = fitz.open(pdf_path)
        total_pages = len(doc)

        try:
            for page_idx in page_indices:
                start_time = time.time()
                page_no = page_idx + 1
                page_folder = os.path.join(run_folder, f"page_{page_no}")
                os.makedirs(page_folder, exist_ok=True)

                try:
                    print(f"[*] 正在处理第 {page_no}/{total_pages} 页...")

                    # 1. 渲染 PDF 页为图片 (300 DPI)
                    img = self._render_page(doc, page_idx)

                    # 2. 版面区域识别 (确定需要哪些模型)
                    layout_engine = OCRModelFactory.get_engine([EngineType.LAYOUT])
                    layout_res = layout_engine.predict(img, layout_nms=True)
                    detected_types = self._get_engine_types(layout_res)

                    if not detected_types:
                        print(f"第 {page_no}/{total_pages} 页未检测到内容，跳过")
                        infos.append({
                            'page': page_no,
                            'status': 'skipped',
                            'execution_time': time.time() - start_time,
                            'detected_types': [],
                            'engine_type': 'N/A',
                            'normalized_items': []
                        })
                        continue

                    generated_images = []
                    print(detected_types)
                    if len(detected_types) == 1:
                        print("只检测到一个区域类型，无需拆分。")
                        generated_images.append({
                            'image': img,
                            'engine_type': list(detected_types)  # OCRLightModelFactory 接收 list
                        })
                    else:
                        print("检测到多个区域类型，需要拆分区域。")
                        merged_regions = self._split_regions(layout_res)
                        generated_images = self.visualize_regions(img, merged_regions, page_folder)

                    engines_used = []
                    page_normalized_items = []
                    for item_idx, item in enumerate(generated_images):
                        engine = item.get("engine_type")
                        image = item.get("image")
                        active_engine = OCRLightModelFactory.get_engine(engine)
                        raw_output = active_engine.predict(image)

                        # 使用标准化结果并按 y 组装
                        normalized_items = self._normalize_results(raw_output, active_engine)
                        page_normalized_items.extend(normalized_items)

                        # 不再单独保存每个识别对象的 json，统一汇总输出到一个 txt
                        _ = item_idx

                        engines_used.append((engine, active_engine))

                    page_normalized_items.sort(key=lambda x: x.get('y', 0))

                    execution_time = time.time() - start_time
                    print("Execution time for page {}: {} seconds".format(page_no, execution_time))

                    info = {
                        'page': page_no,
                        'status': 'ok',
                        'execution_time': execution_time,
                        'detected_types': [eg.value for eg in detected_types],
                        'engine_type': type(engines_used[0][1]).__name__ if engines_used else 'N/A',
                        'normalized_items': page_normalized_items
                    }
                    infos.append(info)

                except Exception as page_error:
                    execution_time = time.time() - start_time
                    print(f"[WARN] 第 {page_no}/{total_pages} 页处理异常，已跳过: {page_error}")
                    infos.append({
                        'page': page_no,
                        'status': 'error',
                        'execution_time': execution_time,
                        'detected_types': [],
                        'engine_type': 'N/A',
                        'error': str(page_error),
                        'normalized_items': []
                    })
        finally:
            doc.close()

        return infos

    def process_document(self, pdf_path: str, output_file: str = "final_result.txt"):
        """主执行流程：PDF拆分 -> 版面分析 -> 路由识别 -> 归一化排序 -> 写入"""
        all_content = []
        doc = fitz.open(pdf_path)
        base_path = r"C:\Users\HUAWEI\Desktop\dyysai\lightModel\test7"
        extension = ".txt"

        # 记录总处理时间的开始时间
        total_start_time = time.time()
        # 用于记录每一页的处理信息
        processing_info = []

        run_folder = os.path.join(base_path, f"run_{int(time.time())}")
        os.makedirs(run_folder, exist_ok=True)

        total_pages = len(doc)
        doc.close()

        # 仅使用多进程并发：按进程数平均拆分 page_idx 给各子进程
        num_processes = min(multiprocessing.cpu_count(), total_pages) if total_pages > 0 else 1
        page_chunks = self._split_page_chunks(total_pages, num_processes)

        with concurrent.futures.ProcessPoolExecutor(max_workers=len(page_chunks) if page_chunks else 1) as process_executor:
            future_to_chunk = {
                process_executor.submit(self._process_page_chunk, pdf_path, chunk, run_folder, extension): chunk
                for chunk in page_chunks
            }

            for future in concurrent.futures.as_completed(future_to_chunk):
                try:
                    chunk_infos = future.result()
                    processing_info.extend(chunk_infos)
                except Exception as chunk_error:
                    # 页块级异常：跳过，不影响其他页块
                    print(f"[ERROR] 页块任务异常，已跳过该块: {chunk_error}")

        processing_info.sort(key=lambda x: x['page'])

        # 接收处理结果，按页排序组装；同页已按 y 排序
        for info in processing_info:
            page_no = info['page']
            for item in info.get('normalized_items', []):
                all_content.append({
                    'page': page_no,
                    'y': item.get('y', 0),
                    'content': item.get('content', '')
                })

        all_content.sort(key=lambda x: (x['page'], x['y']))

        # 将标准化后的组装结果写入一个 txt 文件
        final_path = os.path.join(run_folder, output_file)
        with open(final_path, 'w', encoding='utf-8') as f:
            current_page = None
            for item in all_content:
                if item['page'] != current_page:
                    current_page = item['page']
                    f.write(f"\n===== 第 {current_page} 页 =====\n")
                f.write(f"{item['content']}\n")

        total_end_time = time.time()
        total_execution_time = total_end_time - total_start_time

        info_file_path = os.path.join(run_folder, "processing_info.txt")
        with open(info_file_path, "w", encoding="utf-8") as f:
            f.write("# OCR 处理信息\n\n")
            f.write(f"源文件: {os.path.basename(pdf_path)}\n")
            f.write(f"总页数: {total_pages}\n")
            f.write(f"处理时间: {time.strftime('%Y-%m-%d %H:%M:%S')}\n")
            f.write(f"总执行时间: {total_execution_time:.4f} 秒\n\n")
            f.write("---\n\n")

            for info in processing_info:
                f.write(f"## 第 {info['page']} 页\n")
                f.write(f"状态: {info.get('status', 'ok')}\n")
                f.write(f"执行时间: {info['execution_time']:.4f} 秒\n")
                f.write(f"检测到的类型: {', '.join(info['detected_types'])}\n")
                f.write(f"使用的引擎类型: {info['engine_type']}\n")
                f.write(f"标准化结果条数: {len(info.get('normalized_items', []))}\n")
                if info.get('error'):
                    f.write(f"错误信息: {info['error']}\n")
                f.write("\n")

        print(f"[DONE] 处理信息已保存至: {info_file_path}")
        print(f"[DONE] 组装结果已保存至: {final_path}")
        print(f"[DONE] 总执行时间: {total_execution_time:.4f} 秒")

    def _normalize_results(self, raw_output, engine) -> List[Dict[str, Any]]:
        """适配器：将不同引擎的异构输出统一为排序友好的格式"""
        items = []

        # 情况 A: PaddleOCR 的输出 (你提供的 JSON 结构)
        if isinstance(engine, PaddleOCR):
            for ocr_result in raw_output:
                if not isinstance(ocr_result, dict):
                    continue

                texts = ocr_result.get("rec_texts", [])
                boxes = ocr_result.get("rec_boxes", [])
                # rec_texts 和 rec_boxes 是索引平行的
                for i in range(min(len(texts), len(boxes))):
                    # boxes[i] 为 [x1, y1, x2, y2]
                    items.append({
                        "y": boxes[i][3],
                        "content": texts[i]
                    })

        # 情况 B: PPStructureV3 的输出 (List of Dicts)
        elif isinstance(engine, PPStructureV3):
            for block in raw_output:
                res_list = block['parsing_res_list']

                for res in res_list:
                    bbox = res.block_bbox
                    content = res.block_content

                    if content is None:
                        continue

                    items.append({'y': bbox[3], 'content': content})
        return items

    def _get_engine_types(self, layout_res):
        """解析 LayoutDetection 结果获取 EngineType 列表 以及转换label"""
        types = set()
        # 兼容处理：有些版本返回列表，有些返回带 'boxes' 的字典
        boxes = layout_res[0].get('boxes', []) if isinstance(layout_res, list) and 'boxes' in layout_res[0] else layout_res
        for box in boxes:
            label = box.get('label')
            if label in self.label_map:
                # 将 box 的 label 属性赋值为 self.label_map[label]
                box['label'] = self.label_map[label].value
                types.add(self.label_map[label])
        return types

    def _render_page(self, doc, page_idx):
        page = doc.load_page(page_idx)
        pix = page.get_pixmap(matrix=fitz.Matrix(300 / 72, 300 / 72))
        img = np.frombuffer(pix.samples, dtype=np.uint8).reshape(pix.h, pix.w, pix.n)
        return cv2.cvtColor(img, cv2.COLOR_RGB2BGR)

    def _split_regions(self, layout_res):
        """
        前提条件：存在多种EngineType，等同于有多个region
        拆分区域：根据版面识别结果，处理重叠区域并合并区域
        """

        """
        拆分区域步骤如下：
        1.首先根据layout_res的元素中的boxes的y1坐标从小到大进行排序，相当于从上到下；另外将其中的label转换为label_map对应的EngineType
        2.对排序后的layout_res进行遍历，每一个元素我们称之为区域
        3.开始遍历：判断区域是否存在重叠的情况
        3.1 存在重叠情况：
        3.1.1 如果重叠区域类型相同则合并，范围为上一个区域的y1到这一个区域的y2
        3.1.2 处理小区域被包含在大区域的重叠情况：去除掉小区域，以大区域为准
        3.1.3 未被包含在大区域的重叠情况：如果当前区域的上一个区域和下一个区域为同一类型（EngineType）的区域，则去除当前区域；
        如果不为同一类型，则继续判断下一区域与其下一区域是否重叠，直至不存在重叠，然后将这一组重叠区域进行合并，新的合并区域的类型是这一组区域中面积最大的区域的类型
        3.2 不存在重叠情况：
        3.2.1 当前区域与上一个区域为相同类型：两个区域合并为一个大区域，范围为上一个区域的y1到这一个区域的y2
        3.2.2 当前区域与上一个区域为不同类型：当前区域的范围为上一个区域的y2到当前区域的y2，相当于加入两个区域之间的范围
        3.3 特殊规则：
        3.3.1 第一个区域的范围要加入其头顶的范围，相当于y1=0
        3.3.2 最后一个区域的范围要加入其到脚注的范围，相当于y2=脚注的y坐标
        4.完成遍历：每次遍历时还需要按照对应区域的label映射到EngineType进行分类，得到dict{EngineType, list}
        5.遍历dict{EngineType, list}：for key, value：每一个key生成一幅图像
        5.1 初始化一个白板white
        5.2 遍历value中获取到的区域坐标（y1-y2），到图像img中同样复制对应的区域到white中
        5.3 获取到不同EngineType的图像
        """

        # 通过验证代码获取 'boxes'，确保 layout_res 是一个包含一个元素的列表，且该元素包含 'boxes' 键
        boxes = layout_res[0].get('boxes', []) if isinstance(layout_res, list) and 'boxes' in layout_res[0] else layout_res
        # 将所有区域按 y1 坐标进行排序
        sorted_layout_res = sorted(boxes, key=lambda box: box.get("coordinate")[1])

        merged_regions = []
        current_region = None

        for idx, region in enumerate(sorted_layout_res):
            current_box = region['coordinate']
            # current_type = self.label_map.get(region.get('label'), None)
            current_type = region.get('label', None)
            current_y1, current_y2 = current_box[1], current_box[3]

            # 超过脚注高度进行跳过
            # if(current_y1 > 850):
            #     continue
            # print(current_region)
            # print("\n")
            # print(region)

            if current_region:
                prev_y1, prev_y2 = current_region['coordinate'][1], current_region['coordinate'][3]
                prev_type = current_region['label']

                # 3.1 判断区域是否重叠
                if current_y1 <= prev_y2:
                    # 3.1.1 如果重叠区域类型相同，合并(y1不变，y2设置为最大的)
                    if current_type == prev_type:
                        current_region['coordinate'][3] = max(prev_y2, current_y2)
                        continue  # 继续处理下一个区域
                    # 3.1.2 小区域被包含在大区域，去除小区域
                    elif current_y1 >= prev_y1 and current_y2 <= prev_y2:
                        continue
                    # 3.1.3 不同类型区域，检查是否下一个区域也属于相同类型
                    else:
                        if idx + 1 < len(sorted_layout_res):
                            next_region = sorted_layout_res[idx + 1]
                            if current_type == next_region.get('label'):
                                continue  # 合并当前区域与下一个区域
                            else:
                                merged_regions.append(current_region)  # 添加当前区域
                                current_region = region  # 开始新区域
                                continue
                else:  # 不存在重叠
                    if current_type == prev_type:  # 合并同类型区域
                        current_region['coordinate'][3] = max(prev_y2, current_y2)
                    else:
                        merged_regions.append(current_region)  # 将上一个区域添加
                        current_region = region  # 进入新区域
                        region['coordinate'][1] = min(prev_y2, current_y1)  # 将两个区域的空白处合并给当前区域
            else:
                # region['coordinate'][1] = 0  # 第一个区域的 y1 设为 0
                current_region = region

        # 4. 特殊规则：头顶与脚注的处理
        merged_regions.append(current_region)
        if merged_regions:
            merged_regions[0]['coordinate'][1] = 0  # 第一个区域的 y1 设为 0
            # last_region = merged_regions[-1]
            # last_region['coordinate'][3] = 850  # 最后一个区域的 y2 设为脚注 y2，确保该值与页面底部一致

        return merged_regions

    def visualize_regions(self, img, merged_regions, output_path):
        """可视化合并后的区域并保存到本地，同时返回所有生成的图像"""
        # 存储所有生成的图像
        generated_images = []
        # 按照 EngineType 划分区域
        regions_by_type = {}
        for region in merged_regions:
            region_type = region.get('label', 'Unknown')
            if region_type not in regions_by_type:
                regions_by_type[region_type] = []
            regions_by_type[region_type].append(region)

        for region_type, regions in regions_by_type.items():
            img_copy = np.ones_like(img) * 255

            for region in regions:
                # 使用 coordinate 来获取坐标
                coordinate = region.get('coordinate')

                if not coordinate:
                    print(f"跳过无效区域: {region}")
                    continue

                # 确保 y1, y2 是整数类型
                y1, y2 = int(coordinate[1]), int(coordinate[3])
                # 设置 x1 为 0，表示矩形框的左侧
                x1 = 0
                # 使用图像的宽度作为矩形框的右侧
                x2 = img.shape[1]
                # 获取原图中对应区域的图像
                region_img = img[y1:y2, x1:x2]
                # 将该区域复制到空白画板上的相同位置
                img_copy[y1:y2, x1:x2] = region_img

            # 保存为不同的图像
            output_file_path = os.path.join(output_path, f"{region_type}_regions.png")
            cv2.imwrite(output_file_path, img_copy)
            print(f"可视化结果已保存到: {output_file_path}")

            # 将生成的图像和其对应的 EngineType 添加到列表中
            generated_images.append({
                'image': img_copy,
                'engine_type': [EngineType.from_value(region_type)]
            })

        return generated_images

    def _split_or_not(self, detected_types, layout_res):
        """根据 detected_types 判断是否需要拆分区域"""
        if len(detected_types) == 1:
            # 如果只有一个元素，则无需拆分区域，直接返回
            print("只检测到一个区域类型，无需拆分。")
            return layout_res
        else:
            # 如果多个元素，则根据 layout_res 拆分区域
            print("检测到多个区域类型，需要拆分区域。")
            return self._split_regions(layout_res)  # 使用已有的 _split_regions 方法进行拆分


def convert_to_serializable(obj):
    if isinstance(obj, np.integer):
        return int(obj)
    if isinstance(obj, np.floating):
        return float(obj)
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    if isinstance(obj, dict):
        return {k: convert_to_serializable(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [convert_to_serializable(i) for i in obj]
    return obj


# --- 启动 ---
if __name__ == "__main__":
    # 使用你之前定义的 OCRModelFactory
    processor = DocumentOCRSystemV4(output_dir=r"C:\Users\HUAWEI\Desktop\dyysai\output\md")
    processor.process_document(r"C:\Users\HUAWEI\Desktop\dyysai\test7.pdf")
