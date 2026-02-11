from paddleocr import PPStructureV3, PaddleOCR
from paddleocr import LayoutDetection
from enum import Enum, auto


class EngineType(Enum):
    """
    auto() 方法自动为枚举值进行赋值，保证具体的值是唯一的和避免了手动赋值
    """
    LAYOUT = "layout"  # 纯版面分析
    TEXT = "text"  # 纯文字识别 (Det + Rec)
    TABLE = "table"  # 表格识别 (Structure + Det + Rec)
    FORMULA = "formula"

    @staticmethod
    def from_value(value):
        """
        根据值获取对应的 EngineType 对象
        :param value: 枚举值
        :return: 对应的 EngineType 对象
        """
        for engine_type in EngineType:
            if engine_type.value == value:
                return engine_type
        raise ValueError(f"Invalid value '{value}' for EngineType")

"""OCR 模型单例工厂"""
class OCRModelFactory:
    """
    类变量即静态变量： Python 类中定义的变量（非 self. 变量）默认就是全局共享的单例状态。
    """
    _instances = {}

    @classmethod
    def get_engine(cls, engine_types: list[EngineType], **kwargs):
        """
        根据类型列表获取模型实例，如果不存在则初始化，存在则直接返回
        """
        # 验证输入：LAYOUT只能单独出现
        if EngineType.LAYOUT in engine_types and len(engine_types) > 1:
            raise ValueError("LAYOUT只能单独出现，不能与其他类型组合")
        
        # 生成缓存键：基于排序后的类型名称元组
        cache_key = tuple(sorted([et.name for et in engine_types]))
        
        if cache_key not in cls._instances:
            print(f"[Init] 正在初始化引擎组合: {', '.join([et.name for et in engine_types])}...")
            cls._instances[cache_key] = cls._create_engine(engine_types, **kwargs)
        return cls._instances[cache_key]

    @staticmethod
    def _create_engine(engine_types: list[EngineType], **kwargs):
        # 默认通用配置
        # base_config = {
        #     "lang": 'ch',
        #     **kwargs
        # }

        # 处理LAYOUT单独出现的情况
        if len(engine_types) == 1 and engine_types[0] == EngineType.LAYOUT:
            # 只做版面检测，关闭文字和表格识别
            return LayoutDetection(model_name="PP-DocLayout-M")

        # 处理仅有文本的情况
        # if len(engine_types) == 1 and engine_types[0] == EngineType.TEXT:
            # 纯 OCR 产线，关闭版面分析
            # return PaddleOCR(
            #             ocr_version = "PP-OCRv4",
            #             use_doc_orientation_classify=True,
            #             use_doc_unwarping=False,
            #             use_textline_orientation=False,
            #             format_block_content=True,
            #             lang="ch"
            #         )

        # 处理表格和/或公式的情况
        use_text = EngineType.TEXT in engine_types
        use_table = EngineType.TABLE in engine_types
        use_formula = EngineType.FORMULA in engine_types

        # 如果文本和表格或者公式同时存在时，直接使用表格或公式的PPStructureV3
        return PPStructureV3(
            device="cpu",
            formula_recognition_model_name="PP-FormulaNet_plus-L" if use_formula else None,
            text_detection_model_name="PP-OCRv5_mobile_det",
            text_recognition_model_name="PP-OCRv5_mobile_rec",
            use_doc_orientation_classify=False,
            use_doc_unwarping=False,
            use_textline_orientation=False,
            use_table_recognition=use_table,
            use_formula_recognition=use_formula,
            use_chart_recognition=False,
            use_seal_recognition=False,
            use_region_detection=False
        )


