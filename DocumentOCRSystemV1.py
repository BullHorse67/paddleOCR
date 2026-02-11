import os
import time

import fitz
import cv2
import numpy as np
from typing import List, Dict, Any

from paddleocr import PaddleOCR

from OCRModelFactory import OCRModelFactory, EngineType

# 假设 OCRModelFactory 和 EngineType 已定义

"""
V1版本：
paddle自带save_to_markdown方法保存为markdown格式
这会导致多个md文件生成以及生成表格的相关图片
在结束时进行了手动删除
"""

class DocumentOCRSystemV1:
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

    def process_document(self, pdf_path: str, output_file: str = "final_result.md"):
        """主执行流程：PDF拆分 -> 版面分析 -> 路由识别 -> 归一化排序 -> 写入"""
        all_content = []
        to_markdown_list = []
        doc = fitz.open(pdf_path)
        temp_markdown_files = []

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

            # 3. 动态获取模型组合并进行预测
            active_engine = OCRModelFactory.get_engine(list(detected_types))
            # 统一调用 predict
            raw_output = active_engine.predict(img)
            
            # 为当前页生成临时markdown文件
            temp_markdown_path = os.path.join(self.output_dir, f"temp_page_{page_idx + 1}.md")
            for output in raw_output:
                output.save_to_markdown(save_path=temp_markdown_path)
            
            # 保存临时文件路径
            temp_markdown_files.append(temp_markdown_path)

            # to_markdown_list.extend(raw_output)

            # save_path = r"C:\Users\HUAWEI\Desktop\dyysai\table\pdf"
            # for res in raw_output:
            #     res.save_to_json(save_path=save_path)

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
            print("Execution time:", end_time - start_time, "seconds")

        # 7. 写入单个文件
        # final_path = os.path.join(self.output_dir, output_file)
        # with open(final_path, "w", encoding="utf-8") as f:
        #     f.write("\n\n".join(all_content))

        # for item in to_markdown_list:
        #     item.save_to_markdown(save_path=r"C:\Users\HUAWEI\Desktop\dyysai\output\test.md")

        # 合并所有临时markdown文件为一个文件
        final_path = os.path.join(self.output_dir, output_file)
        
        # 确保输出目录存在
        if not os.path.exists(self.output_dir):
            os.makedirs(self.output_dir)
        
        # 创建并写入最终的markdown文件
        with open(final_path, "w", encoding="utf-8") as final_file:
            for i, temp_file_path in enumerate(temp_markdown_files):
                # 添加页码标题
                final_file.write(f"# 第 {i + 1} 页\n\n")
                
                # 读取并写入临时文件内容
                if os.path.exists(temp_file_path):
                    with open(temp_file_path, "r", encoding="utf-8") as temp_file:
                        final_file.write(temp_file.read())
                    final_file.write("\n\n")

        # 删除临时markdown文件
        for temp_file_path in temp_markdown_files:
            if os.path.exists(temp_file_path):
                os.remove(temp_file_path)
                print(f"已删除临时文件: {temp_file_path}")
        
        # 清理自动生成的不需要的文件和目录
        self._cleanup_generated_files()

        doc.close()
        print(f"[DONE] 结果已保存至: {final_path}")

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
        else:
            for block in raw_output:
                res_list = block['parsing_res_list']

                for res in res_list:
                    bbox = res.bbox
                    content = res.content

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
    
    def _cleanup_generated_files(self):
        """
        清理save_to_markdown方法自动生成的不需要的文件和目录
        主要删除imgs目录及其内容
        """
        # 检查并删除imgs目录
        imgs_dir = os.path.join(self.output_dir, "imgs")
        if os.path.exists(imgs_dir) and os.path.isdir(imgs_dir):
            # 删除目录中的所有文件
            for root, dirs, files in os.walk(imgs_dir):
                for file in files:
                    file_path = os.path.join(root, file)
                    try:
                        os.remove(file_path)
                        print(f"已删除: {file_path}")
                    except Exception as e:
                        print(f"删除文件 {file_path} 时出错: {e}")
            
            # 删除空目录
            try:
                os.rmdir(imgs_dir)
                print(f"已删除目录: {imgs_dir}")
            except Exception as e:
                print(f"删除目录 {imgs_dir} 时出错: {e}")


def ensure_dict(obj) -> Dict:
    """
    【核心工具】将任何对象（尤其是 LayoutParsingBlock）强制转换为字典
    """
    if isinstance(obj, dict):
        return obj
    # 尝试使用 vars() 获取对象的属性字典
    try:
        return vars(obj)
    except TypeError:
        # 如果 vars 不行，说明可能是 C 扩展对象或定义了 __slots__
        # 这种情况下尝试读取所有非私有属性
        return {attr: getattr(obj, attr) for attr in dir(obj) if not attr.startswith('_')}

# --- 启动 ---
if __name__ == "__main__":
    # 使用你之前定义的 OCRModelFactory
    processor = DocumentOCRSystemV1(output_dir=r"C:\Users\HUAWEI\Desktop\dyysai\output\md")
    processor.process_document(r"C:\Users\HUAWEI\Desktop\dyysai\test2.pdf")