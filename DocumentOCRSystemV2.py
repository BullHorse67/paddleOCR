import json
import os
import time

import fitz
import cv2
import numpy as np
from typing import List, Dict, Any
from paddleocr import PaddleOCR, PPStructureV3
from OCRLightModelFactory import OCRLightModelFactory
from OCRModelFactory import OCRModelFactory, EngineType

class DocumentOCRSystemV2:
    def __init__(self, output_dir: str):
        self.output_dir = output_dir
        if not os.path.exists(output_dir):
            os.makedirs(output_dir)

        self.label_map = {
            'text': EngineType.TEXT,
            'number': EngineType.TEXT,
            'paragraph_title': EngineType.TEXT,
            'figure_title': EngineType.TEXT,
            'table': EngineType.TABLE,
            'formula': EngineType.FORMULA
        }

    def process_document(self, pdf_path: str, output_file: str = "final_result.txt"):
        """主执行流程：PDF拆分 -> 版面分析 -> 路由识别 -> 归一化排序 -> 写入"""
        all_content = []
        doc = fitz.open(pdf_path)
        base_path = r"C:\Users\HUAWEI\Desktop\dyysai\lightModel"
        extension = ".txt"
        
        # 记录总处理时间的开始时间
        total_start_time = time.time()
        
        # 用于记录每一页的处理信息
        processing_info = []

        for page_idx in range(len(doc)):
            start_time = time.time()
            print(f"[*] 正在处理第 {page_idx + 1}/{len(doc)} 页...")

            # 1. 渲染 PDF 页为图片 (300 DPI)
            img = self._render_page(doc, page_idx)

            # 2. 版面区域识别 (确定需要哪些模型)
            layout_engine = OCRModelFactory.get_engine([EngineType.LAYOUT])
            layout_res = layout_engine.predict(img, layout_nms=True)
            detected_types = self._get_engine_types(layout_res)

            if not detected_types:
                continue

            print(detected_types)

            # 3. 动态获取模型组合并进行预测
            # active_engine = OCRModelFactory.get_engine(list(detected_types))
            active_engine = OCRLightModelFactory.get_engine(list(detected_types))
            # 统一调用 predict
            raw_output = active_engine.predict(img)

            for r in raw_output:
                # 保存为JSON
                file_name = f"{page_idx}{extension}"
                save_path = os.path.join(base_path, file_name)
                data = r._to_json()
                # data = r._to_markdown()
                converted_data = convert_to_serializable(data)
                with open(save_path, 'w', encoding='utf-8') as f:
                    json.dump(converted_data, f, ensure_ascii=False, indent=4)
                # r.save_to_json(save_path=r"C:\Users\HUAWEI\Desktop\dyysai\lightModel\test4")

            # 4. 结果归一化 (核心修改：处理 PaddleOCR JSON 和 PPStructure Dict)
            # normalized_items = self._normalize_results(raw_output, active_engine)

            # 5. 根据 y 坐标排序 (从上至下)
            # normalized_items.sort(key=lambda x: x['y'])

            # 6. 提取纯净内容
            # page_content = [f"--- Page {page_idx + 1} ---"]
            # for item in normalized_items:
            #     page_content.append(item['content'])
            #
            # all_content.append("\n".join(page_content))

            end_time = time.time()
            execution_time = end_time - start_time
            print("Execution time:", execution_time, "seconds")
            
            # 记录处理信息
            info = {
                'page': page_idx + 1,
                'execution_time': execution_time,
                'detected_types': [et.name for et in detected_types],
                'engine_type': type(active_engine).__name__
            }
            processing_info.append(info)

        # 计算总处理时间
        total_end_time = time.time()
        total_execution_time = total_end_time - total_start_time
        
        # 7. 写入单个文件
        # final_path = os.path.join(self.output_dir, output_file)
        # with open(final_path, "w", encoding="utf-8") as f:
        #     f.write("\n\n".join(all_content))
        
        # 写入处理信息到txt文件
        info_file_path = os.path.join(base_path, "processing_info.txt")
        with open(info_file_path, "w", encoding="utf-8") as f:
            f.write("# OCR 处理信息\n\n")
            f.write(f"源文件: {os.path.basename(pdf_path)}\n")
            f.write(f"总页数: {len(doc)}\n")
            f.write(f"处理时间: {time.strftime('%Y-%m-%d %H:%M:%S')}\n")
            f.write(f"总执行时间: {total_execution_time:.4f} 秒\n\n")
            f.write("---\n\n")
            
            for info in processing_info:
                f.write(f"## 第 {info['page']} 页\n")
                f.write(f"执行时间: {info['execution_time']:.4f} 秒\n")
                f.write(f"检测到的类型: {', '.join(info['detected_types'])}\n")
                f.write(f"使用的引擎类型: {info['engine_type']}\n\n")

        doc.close()
        # print(f"[DONE] 结果已保存至: {final_path}")
        print(f"[DONE] 处理信息已保存至: {info_file_path}")
        print(f"[DONE] 总执行时间: {total_execution_time:.4f} 秒")

    def _normalize_results(self, raw_output, engine) -> List[Dict[str, Any]]:
        """适配器：将不同引擎的异构输出统一为排序友好的格式"""
        items = []

        # 情况 A: PaddleOCR 的输出 (你提供的 JSON 结构)
        if isinstance(engine, PaddleOCR):
            for ocr_result in raw_output:
                if not isinstance(ocr_result, dict): continue

                texts = ocr_result.get("rec_texts", [])
                boxes = ocr_result.get("rec_boxes", [])
                # rec_texts 和 rec_boxes 是索引平行的
                for i in range(len(texts)):
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
                    # bbox = res.bbox
                    bbox = res.block_bbox
                    # content = res.content
                    content = res.block_content

                    if content is None : continue

                    items.append({'y': bbox[3], 'content': content})

                # if r_type == 'table':
                #     # table 的 res 可能是字典或对象
                #     html = res.get('html', '') if isinstance(res, dict) else getattr(res, 'html', '')
                #     content = f"[TABLE]\n{html}\n[TABLE_END]"
                # elif r_type == 'formula':
                #     content = f"[FORMULA]: {res}"
                # elif r_type in ['text', 'title', 'plain_text']:
                #     if isinstance(res, list):
                #         # res 内部可能又是 LayoutParsingBlock 列表
                #         lines = []
                #         for line in res:
                #             txt = line.get('text', '') if isinstance(line, dict) else getattr(line, 'text', '')
                #             if txt: lines.append(txt)
                #         content = "\n".join(lines)
                #     else:
                #         content = str(res)
        return items

    def _extract_structure_content(self, region: Dict) -> str:
        """从 PPStructure 区域中提取干净的内容"""
        r_type = region.get('type')
        res = region.get('res', [])

        if r_type == 'table':
            # 返回表格 HTML
            return f"[TABLE]\n{res.get('html', '')}\n[TABLE_END]"
        elif r_type == 'formula':
            # 返回 LaTeX 公式
            return f"[FORMULA]: {res}"
        elif r_type in ['text', 'paragraph_title', 'figure_title']:
            # 如果是文本块，合并内部所有行
            if isinstance(res, list):
                return "\n".join([line.get('text', '') for line in res])
        return ""

    def _get_engine_types(self, layout_res):
        """解析 LayoutDetection 结果获取 EngineType 列表"""
        types = set()
        # 兼容处理：有些版本返回列表，有些返回带 'boxes' 的字典
        boxes = layout_res[0].get('boxes', []) if isinstance(layout_res, list) and 'boxes' in layout_res[
            0] else layout_res
        for box in boxes:
            label = box.get('label')
            if label in self.label_map:
                types.add(self.label_map[label])
        return types

    def _render_page(self, doc, page_idx):
        page = doc.load_page(page_idx)
        pix = page.get_pixmap(matrix=fitz.Matrix(300 / 72, 300 / 72))
        img = np.frombuffer(pix.samples, dtype=np.uint8).reshape(pix.h, pix.w, pix.n)
        return cv2.cvtColor(img, cv2.COLOR_RGB2BGR)

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
    processor = DocumentOCRSystemV2(output_dir=r"C:\Users\HUAWEI\Desktop\dyysai\output\md")
    processor.process_document(r"C:\Users\HUAWEI\Desktop\dyysai\test4.pdf")