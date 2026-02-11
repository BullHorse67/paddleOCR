# PaddleOCR PDF 文档识别系统

本项目基于 **PaddleOCR / PP-Structure** 构建，面向 PDF 文档识别场景，核心目标是：

- 先做版面分析（Layout）
- 再根据页面内容类型进行模型路由（文本 / 表格 / 公式）
- 最终将跨页识别结果按阅读顺序归一化输出为文本

项目核心由两个文件驱动：

- `DocumentOCRSystemV4.py`：文档处理主流程（并发处理、页面拆分、区域路由、结果归一化）
- `OCRLightModelFactory.py`：轻量模型工厂（按 EngineType 组合创建并缓存对应引擎）

同时，项目还包含 `OCRModelFactory.py`，可作为另一种工厂模式（偏通用的 PPStructureV3 组合）与轻量工厂形成双工厂设计。

---

## 1. 项目能力概览

### 1.1 支持的识别类型

通过 `EngineType` 枚举统一表达：

- `layout`：版面检测
- `text`：文本识别
- `table`：表格识别
- `formula`：公式识别

`DocumentOCRSystemV4` 中的 `label_map` 会把版面检测出的标签（如 `paragraph_title`、`table`、`formula`）映射到上述引擎类型，从而驱动后续模型选择。

### 1.2 处理流程（高层）

1. 读取 PDF，按 CPU 数量把页码均匀切分为多个连续页块。
2. 多进程并发处理每个页块（每页异常不影响其它页）。
3. 单页先渲染为图像（300 DPI）。
4. 用布局模型识别区域，判断该页涉及哪些类型（文本/表格/公式）。
5. 若仅一种类型：整页直接走对应 OCR 引擎。
6. 若多种类型：先区域合并，再按类型生成分区图像后分别识别。
7. 识别结果统一标准化为 `{y, content}`，并按页面+纵坐标排序。
8. 输出：
   - `final_result.txt`：最终全文拼接结果
   - `processing_info.txt`：每页状态、耗时、检测类型、错误信息等

---

## 2. 核心文件说明

## 2.1 `DocumentOCRSystemV4.py`

该类是整体编排器，负责“并发调度 + 版面分析 + 路由识别 + 结果落盘”。

### 2.1.1 并发与容错

- 使用 `ProcessPoolExecutor` 做多进程页块并发。
- 通过 `_split_page_chunks` 保证页码切分连续且覆盖完整。
- 页面级 try/except：某一页失败后记录错误并跳过，不阻断全任务。
- 页块级异常也被捕获，避免整体任务中断。

### 2.1.2 路由识别策略

- 先使用 `OCRModelFactory.get_engine([EngineType.LAYOUT])` 获取布局检测引擎。
- `_get_engine_types` 解析布局输出并收集该页所需 `EngineType` 集合。
- 当页面只有一种类型时，直接整页识别。
- 多类型页面会调用 `_split_regions` + `visualize_regions` 做区域重组后分类型识别。

### 2.1.3 结果归一化

通过 `_normalize_results` 对不同引擎输出结构做统一：

- `PaddleOCR` 输出：读取 `rec_texts` 与 `rec_boxes`，用框的 y 坐标做排序键。
- `PPStructureV3` 输出：提取 `parsing_res_list` 中每个块的 `block_bbox` 与 `block_content`。

归一化后统一为：

```text
{"y": <number>, "content": <string>}
```

再按 `(page, y)` 排序，写入总结果文件。

---

## 2.2 `OCRLightModelFactory.py`

该文件实现“轻量模型单例工厂”，用于按页面类型动态拿到最合适模型，并通过缓存避免重复初始化。

### 2.2.1 核心设计

- `get_engine(engine_types)`：
  - 校验 `LAYOUT` 不能和其他类型混用。
  - 用“排序后的类型名元组”作为缓存键。
  - 命中缓存直接复用，未命中则创建。

- `_create_engine(engine_types)`：
  - `[LAYOUT]`：`LayoutDetection(PP-DocLayout-M)`
  - `[TEXT]`：`PaddleOCR(PP-OCRv4)`（轻量文本 OCR 配置）
  - 含 `FORMULA`：`PPStructureV3` + 公式模型 `PP-FormulaNet_plus-L`
  - 含 `TABLE` 且不含公式：`PPStructureV3`（开启表格识别）

### 2.2.2 为什么叫“轻量工厂”

从实现上看，轻量工厂在文本场景直接使用 `PaddleOCR`，在结构化场景使用移动端检测/识别模型配置（如 `PP-OCRv5_mobile_det/rec`），兼顾速度与资源占用，适合文档批处理中的按页动态路由。

---

## 3. 双工厂模式说明

你当前项目本质上提供了两种模型工厂路径：

1. **`OCRModelFactory`（通用工厂）**
   - 更偏统一的 `PPStructureV3` 方案，适合全功能场景。
2. **`OCRLightModelFactory`（轻量工厂）**
   - 按类型细分，文本场景可直接走 `PaddleOCR`，结构场景走 `PPStructureV3` 轻量配置。

在 `DocumentOCRSystemV4` 中：

- 布局检测阶段使用 `OCRModelFactory` 获取 `LAYOUT` 引擎。
- 页面识别阶段使用 `OCRLightModelFactory` 根据区域类型组合获取引擎。

这种组合方式能在“识别覆盖能力”与“执行效率”之间取得平衡。

---

## 4. 目录与关键文件

```text
.
├── DocumentOCRSystemV4.py      # 核心流程：并发处理 + 版面路由 + 结果汇总
├── OCRLightModelFactory.py     # 轻量模型工厂（单例缓存）
├── OCRModelFactory.py          # 通用模型工厂（EngineType 定义）
└── README.md                   # 项目说明（本文档）
```

---

## 5. 最小使用示例

> 说明：以下示例对应当前代码中的类接口，路径请替换为你的本地文件路径。

```python
from DocumentOCRSystemV4 import DocumentOCRSystemV4

processor = DocumentOCRSystemV4(output_dir=r"./output")
processor.process_document(pdf_path=r"./demo.pdf", output_file="final_result.txt")
```

运行后会在任务目录下生成：

- `final_result.txt`：按页分节的识别文本
- `processing_info.txt`：每页处理详情

---

## 6. 适用场景

- 扫描版/电子版 PDF 的批量 OCR 提取
- 含文本 + 表格 + 公式的混合文档解析
- 需要可追踪处理日志（页级状态、耗时、错误信息）的离线处理任务

---

## 7. 后续可优化建议

1. 将 `process_document` 中硬编码路径（`base_path`）改为配置项或构造参数。
2. 为 `_split_regions` 增加更系统的单元测试，覆盖复杂重叠区域规则。
3. 输出格式可扩展为 JSON/Markdown，便于下游结构化消费。
4. 为不同模型工厂增加可选策略参数（精度优先 / 速度优先）。

---

## 8. 说明

本 README 基于你指定的两个核心文件（`DocumentOCRSystemV4.py` 与 `OCRLightModelFactory.py`）及项目现有代码组织整理，用于帮助快速理解该 PDF OCR 系统的核心机制与双工厂设计思路。
