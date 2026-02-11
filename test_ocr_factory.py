from OCRModelFactory import OCRModelFactory, EngineType

# 测试场景1: 单独的LAYOUT
print("测试场景1: 单独的LAYOUT")
try:
    layout_engine = OCRModelFactory.get_engine([EngineType.LAYOUT])
    print(f"成功创建LAYOUT引擎: {type(layout_engine).__name__}")
except Exception as e:
    print(f"错误: {e}")

# 测试场景2: 单独的TEXT
print("\n测试场景2: 单独的TEXT")
try:
    text_engine = OCRModelFactory.get_engine([EngineType.TEXT])
    print(f"成功创建TEXT引擎: {type(text_engine).__name__}")
except Exception as e:
    print(f"错误: {e}")

# 测试场景3: 单独的TABLE
print("\n测试场景3: 单独的TABLE")
try:
    table_engine = OCRModelFactory.get_engine([EngineType.TABLE])
    print(f"成功创建TABLE引擎: {type(table_engine).__name__}")
except Exception as e:
    print(f"错误: {e}")

# 测试场景4: 单独的FORMULA
print("\n测试场景4: 单独的FORMULA")
try:
    formula_engine = OCRModelFactory.get_engine([EngineType.FORMULA])
    print(f"成功创建FORMULA引擎: {type(formula_engine).__name__}")
except Exception as e:
    print(f"错误: {e}")

# 测试场景5: TEXT + TABLE 组合
print("\n测试场景5: TEXT + TABLE 组合")
try:
    text_table_engine = OCRModelFactory.get_engine([EngineType.TEXT, EngineType.TABLE])
    print(f"成功创建TEXT+TABLE引擎: {type(text_table_engine).__name__}")
except Exception as e:
    print(f"错误: {e}")

# 测试场景6: TEXT + FORMULA 组合
print("\n测试场景6: TEXT + FORMULA 组合")
try:
    text_formula_engine = OCRModelFactory.get_engine([EngineType.TEXT, EngineType.FORMULA])
    print(f"成功创建TEXT+FORMULA引擎: {type(text_formula_engine).__name__}")
except Exception as e:
    print(f"错误: {e}")

# 测试场景7: TABLE + FORMULA 组合
print("\n测试场景7: TABLE + FORMULA 组合")
try:
    table_formula_engine = OCRModelFactory.get_engine([EngineType.TABLE, EngineType.FORMULA])
    print(f"成功创建TABLE+FORMULA引擎: {type(table_formula_engine).__name__}")
except Exception as e:
    print(f"错误: {e}")

# 测试场景8: TEXT + TABLE + FORMULA 组合
print("\n测试场景8: TEXT + TABLE + FORMULA 组合")
try:
    all_engine = OCRModelFactory.get_engine([EngineType.TEXT, EngineType.TABLE, EngineType.FORMULA])
    print(f"成功创建TEXT+TABLE+FORMULA引擎: {type(all_engine).__name__}")
except Exception as e:
    print(f"错误: {e}")

# 测试场景9: 错误场景 - LAYOUT + TEXT 组合
print("\n测试场景9: 错误场景 - LAYOUT + TEXT 组合")
try:
    invalid_engine = OCRModelFactory.get_engine([EngineType.LAYOUT, EngineType.TEXT])
    print(f"错误: 应该抛出异常")
except ValueError as e:
    print(f"成功捕获预期异常: {e}")
except Exception as e:
    print(f"错误: 捕获到意外异常: {e}")

# 测试场景10: 验证实例复用
print("\n测试场景10: 验证实例复用")
try:
    text_engine1 = OCRModelFactory.get_engine([EngineType.TEXT])
    text_engine2 = OCRModelFactory.get_engine([EngineType.TEXT])
    print(f"TEXT引擎复用: {text_engine1 is text_engine2}")
    
    table_engine1 = OCRModelFactory.get_engine([EngineType.TABLE])
    table_engine2 = OCRModelFactory.get_engine([EngineType.TABLE])
    print(f"TABLE引擎复用: {table_engine1 is table_engine2}")
    
    text_table_engine1 = OCRModelFactory.get_engine([EngineType.TEXT, EngineType.TABLE])
    text_table_engine2 = OCRModelFactory.get_engine([EngineType.TABLE, EngineType.TEXT])  # 顺序不同
    print(f"TEXT+TABLE引擎复用(顺序不同): {text_table_engine1 is text_table_engine2}")
except Exception as e:
    print(f"错误: {e}")
