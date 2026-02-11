# DocumentOCRSystemV4 设计思路与问题分析

## 一、整体思路（当前实现）

1. **按页处理 PDF**：
   - 用 `fitz` 打开 PDF，逐页渲染成 300 DPI 的图像。
2. **版面检测决定路由**：
   - 先用 `LAYOUT` 引擎做版面分析，提取每页出现的区域类型（文本/表格/公式等）。
3. **按类型拆分或不拆分**：
   - 如果只检测到一种类型，整页走该类型模型。
   - 如果有多种类型，先做区域合并，再把同类型区域“贴回白底图”形成多张按类型路由的图片。
4. **页内并发识别**：
   - 每页内部使用 `ThreadPoolExecutor(max_workers=4)`，对不同路由图片并发预测。
5. **页间并发处理**：
   - 使用 `ProcessPoolExecutor(max_workers=cpu_count)` 并发处理每一页。
6. **输出信息**：
   - 为每页写识别结果和运行信息，最终汇总到 `processing_info.txt`。

---

## 二、主要风险点

### 1) 进程池调用嵌套函数会触发序列化问题
`process_single_page` 定义在 `process_document` 内部，`ProcessPoolExecutor` 在 Windows / spawn 模式下需要可 pickle 的顶层函数，嵌套函数通常无法被 pickle。

**影响**：可能直接在提交任务时失败，报“can't pickle local object ...process_single_page”。

**建议**：
- 将 `process_single_page` 提升为模块级函数，或改为类的 `@staticmethod` + 显式参数传递；
- 或取消进程池，仅保留线程池（若模型本身释放 GIL 且线程安全）。

### 2) 在多进程中共享 `doc`（fitz.Document）存在对象不可序列化/跨进程不安全风险
当前子任务依赖外层 `doc` 对象；该对象通常不适合跨进程传递。

**建议**：
- 子进程仅传 `pdf_path` + `page_idx`，在子进程内单独 `fitz.open(pdf_path)` 并读取对应页。

### 3) 每页都重新构建 layout/model engine，初始化开销大
每个页面调用 `OCRModelFactory.get_engine([EngineType.LAYOUT])`，若底层会加载模型，开销和显存抖动都很大。

**建议**：
- 进程内做模型单例缓存（懒加载）；
- 页间并发与 GPU 资源要匹配，避免 N 进程争抢同一张卡。

### 4) 双层并发（进程 + 线程）易过度并发
页间 `cpu_count` 个进程、页内固定 4 线程，理论并发上限是 `cpu_count * 4` 个识别任务，可能导致：
- CPU 过载、上下文切换开销大；
- GPU OOM；
- I/O 打满。

**建议**：
- 用统一调度：要么页间并发、页内串行；要么页间少量并发 + 页内少量并发；
- 加资源阈值控制（如总 worker、GPU 队列）。

### 5) 路由类型传参格式可疑
`generated_images` 中 `engine_type` 常是列表（如 `list(detected_types)` 或 `[EngineType.from_value(...)]`），但工厂 `get_engine` 通常期望单个 `EngineType`。

**建议**：
- 明确接口：`engine_type: EngineType`；
- 如果要多类型，拆成多条任务，每条任务绑定单一类型。

### 6) 结果写文件存在覆盖风险
页内线程对同一页使用固定文件名 `f"{page_idx}{extension}"`，多个任务会互相覆盖。

**建议**：
- 文件名加入区域类型/任务 ID/时间戳，如 `f"{page_idx}_{engine.value}_{task_id}.json"`。

### 7) 结果聚合未完成
`all_content` 未使用，`_normalize_results` 已写但未接入主流程，且最终 `output_file` 参数未使用。

**建议**：
- 在线程结果返回时收集 `raw_output`；
- 调用 `_normalize_results` 后按 `(page, y)` 排序；
- 最终统一写入 `output_file`。

### 8) `_split_regions` 会就地修改 `layout_res`
函数会直接改 `region['coordinate']` 和 `label`，可能影响后续复用或调试。

**建议**：
- 进入函数先深拷贝；
- 输出新结构，不修改输入。

### 9) 区域规则与注释目标不完全一致
注释中提到脚注边界、面积最大类型归并等复杂规则，代码里未完整实现（如脚注 y2 固定值逻辑被注释）。

**建议**：
- 将规则拆成可测试的小函数（重叠判定、包含判定、冲突归并策略）；
- 增加单元测试覆盖典型版面样本。

### 10) 路径硬编码为本地 Windows 目录
`base_path`、`__main__` 的输入输出都写死为本地路径，影响可移植性。

**建议**：
- 统一使用 `self.output_dir` 和函数入参；
- 通过 CLI 或配置文件注入路径。

### 11) 异常处理不足
并发任务 `future.result()` 若抛异常会中断整批处理。

**建议**：
- 对每个 future 做异常捕获与记录，保证“单页失败不拖垮全局”。

### 12) 线程安全未验证
`OCRLightModelFactory` 返回的引擎对象若是共享实例，`predict` 是否线程安全未知。

**建议**：
- 明确模型实例生命周期：线程内独占、进程内复用；
- 若不线程安全，改成进程并发或任务串行。

---

## 三、建议的重构方向（实用版）

1. **先做“稳态版本”**：
   - 去掉进程池，仅用页级串行 + 页内小线程池（或反过来）；
   - 确保功能正确、结果可追溯。
2. **再做“受控并发”**：
   - 用 `max_workers=min(页数, 2~4)`，按 GPU/CPU 压测调优。
3. **完成结果主链路**：
   - `predict -> normalize -> page sort -> document merge -> output_file`。
4. **补充可观测性**：
   - 每页记录：模型耗时、队列等待、失败原因、重试次数。
5. **建立回归样例**：
   - 用 3~5 份含文本/表格/公式的 PDF 做基准，比较正确率与耗时。

---

## 四、优先级修复清单（建议）

- P0：修复进程池可运行性（嵌套函数 + `doc` 跨进程问题）。
- P0：修复结果文件覆盖问题。
- P1：接通 `_normalize_results` 与最终 `output_file`。
- P1：清理硬编码路径，改为参数化。
- P2：优化 `_split_regions` 规则完整性与测试覆盖。
- P2：并发模型做压测后再放量。
