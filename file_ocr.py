import os
import tempfile
from PIL import Image
from pdf2image import convert_from_path
from OCRModelFactory import OCRModelFactory, EngineType

# 作为一名资深Python算法工程师，现在你需要搭建一个OCR文件识别系统，文件中存在文本、表格以及公式三种情况，以及其组合的情况
# 实现要求：项目简洁清晰，除了能够明显提升项目效果外的，实现流程中没有提及的尽量不额外增加
# 实现流程：
# 1.所有模型实例通过OCRModelFactory工厂获取
# 2.拆分pdf为单页图像进行遍历处理
# 3.遍历时图像先进行版面区域识别分类，然后根据识别出的类型组成对应的枚举值列表到工厂中获取对应实例
# 4.调用实例模型对图像进行OCR识别
# 5.获取到的结果需要按照原本图像中的顺序进行组装，仅需要提取图像中的文本/表格/公式三种类型的结果，对于其余的坐标信息等不需要返回，并组装成单个文件

# 参照paddleOcr.py的ocr识别过程

def process_pdf(pdf_path, output_md_path):
    """
    处理PDF文件，将结果保存为markdown格式
    
    Args:
        pdf_path: PDF文件路径
        output_md_path: 输出markdown文件路径
    """
    # 将PDF转换为图像
    images = pdf_to_images(pdf_path)
    
    # 处理每一页图像
    markdown_content = []
    for i, image in enumerate(images):
        print(f"处理第 {i+1} 页...")
        # 保存图像为临时文件
        with tempfile.NamedTemporaryFile(suffix='.png', delete=False) as temp_file:
            temp_image_path = temp_file.name
            image.save(temp_image_path)
        
        # 处理单页图像
        page_content = process_image(temp_image_path)
        markdown_content.append(f"# 第 {i+1} 页\n{page_content}")
        
        # 删除临时文件
        os.unlink(temp_image_path)
    
    # 将结果写入markdown文件
    with open(output_md_path, 'w', encoding='utf-8') as f:
        f.write('\n\n'.join(markdown_content))
    
    print(f"处理完成，结果已保存到: {output_md_path}")

def pdf_to_images(pdf_path):
    """
    将PDF文件转换为图像列表
    
    Args:
        pdf_path: PDF文件路径
        
    Returns:
        图像列表
    """
    # 使用pdf2image将PDF转换为图像
    images = convert_from_path(pdf_path)
    return images

def process_image(image_path):
    """
    处理单页图像，识别其中的文本、表格和公式
    
    Args:
        image_path: 图像路径
        
    Returns:
        处理后的markdown内容
    """
    # 先进行版面区域识别分类
    layout_result = perform_layout_analysis(image_path)
    
    # 根据识别出的类型组成对应的枚举值列表
    engine_types = determine_engine_types(layout_result)
    
    # 从工厂中获取对应实例
    ocr_engine = OCRModelFactory.get_engine(engine_types)
    
    # 调用实例模型对图像进行OCR识别
    ocr_result = perform_ocr(ocr_engine, image_path)
    
    # 提取结果并转换为markdown格式
    markdown_content = convert_to_markdown(ocr_result)
    
    return markdown_content

def perform_layout_analysis(image_path):
    """
    进行版面区域识别分类
    
    Args:
        image_path: 图像路径
        
    Returns:
        版面分析结果
    """
    # 获取版面分析引擎
    layout_engine = OCRModelFactory.get_engine([EngineType.LAYOUT])
    
    # 使用版面分析引擎进行分析
    # 参照paddleOcr.py中的get_layout_regions函数
    results = layout_engine.predict(image_path, layout_nms=True)
    
    return results

def determine_engine_types(layout_result):
    """
    根据版面分析结果确定需要使用的引擎类型
    
    Args:
        layout_result: 版面分析结果
        
    Returns:
        EngineType枚举值列表
    """
    engine_types = []
    
    # 分析版面结果，确定需要的引擎类型
    # 参照paddleOcr.py中的get_layout_regions函数
    has_text = False
    has_table = False
    has_formula = False
    
    for res in layout_result:
        try:
            boxes = res['boxes']
        except (TypeError, KeyError):
            # 某些版本中，boxes 实际上就在 res 这一层
            boxes = res
        
        for box in boxes:
            coords = box['coordinate']
            label = box['label']
            score = box['score']
            
            # 过滤低置信度结果
            if score > 0.45:
                label_lower = label.lower()
                if 'text' in label_lower or 'title' in label_lower:
                    has_text = True
                elif 'table' in label_lower:
                    has_table = True
                elif 'formula' in label_lower:
                    has_formula = True
    
    # 根据识别结果添加对应的引擎类型
    if has_table:
        engine_types.append(EngineType.TABLE)
    if has_formula:
        engine_types.append(EngineType.FORMULA)
    if not engine_types and has_text:
        # 只有文本时，添加TEXT类型
        engine_types.append(EngineType.TEXT)
    
    return engine_types

def perform_ocr(ocr_engine, image_path):
    """
    调用OCR引擎对图像进行识别
    
    Args:
        ocr_engine: OCR引擎实例
        image_path: 图像路径
        
    Returns:
        OCR识别结果
    """
    # 参照paddleOcr.py中的paddleOCR和tablePPV3函数
    # 使用OCR引擎进行识别
    result = ocr_engine.predict(image_path, use_layout_detection=True, use_ocr_model=True)
    
    return result

def convert_to_markdown(ocr_result):
    """
    将OCR识别结果转换为markdown格式
    
    Args:
        ocr_result: OCR识别结果
        
    Returns:
        markdown格式的内容
    """
    markdown_parts = []
    
    # 参照paddleOcr.py中的结果处理方式
    # 处理识别结果
    for res in ocr_result:
        # 处理文本结果
        if res['type'] == 'text' or res['type'] == 'title':
            text_content = res['res']['text']
            markdown_parts.append(text_content)
        # 处理表格结果
        elif res['type'] == 'table':
            table_content = convert_table_to_markdown(res['res'])
            if table_content:
                markdown_parts.append(table_content)
        # 处理公式结果
        elif res['type'] == 'formula':
            formula_content = f"$${res['res']['text']}$$"
            markdown_parts.append(formula_content)
    
    return '\n\n'.join(markdown_parts)

def convert_table_to_markdown(table_result):
    """
    将表格结果转换为markdown格式
    
    Args:
        table_result: 表格识别结果
        
    Returns:
        markdown格式的表格
    """
    # 参照paddleOcr.py中的表格处理方式
    # 从表格结果中提取HTML，然后转换为markdown
    html_content = table_result.get('html', '')
    if not html_content:
        return ''
    
    # 这里简化处理，实际项目中可能需要更复杂的HTML转markdown逻辑
    # 或者直接使用表格的文本内容
    table_text = table_result.get('text', '')
    if not table_text:
        return ''
    
    # 简单处理表格文本为markdown格式
    lines = table_text.strip().split('\n')
    if not lines:
        return ''
    
    # 构建markdown表格
    markdown_table = []
    
    # 添加表头
    header = '| ' + ' | '.join(lines[0].split('\t')) + ' |'
    separator = '| ' + ' | '.join(['---' for _ in lines[0].split('\t')]) + ' |'
    markdown_table.append(header)
    markdown_table.append(separator)
    
    # 添加表格内容
    for line in lines[1:]:
        if line.strip():
            row_content = '| ' + ' | '.join(line.split('\t')) + ' |'
            markdown_table.append(row_content)
    
    return '\n'.join(markdown_table)

# 示例用法
if __name__ == "__main__":
    # 替换为实际的PDF文件路径和输出markdown文件路径
    pdf_path = "example.pdf"
    output_md_path = "output.md"
    process_pdf(pdf_path, output_md_path)
