# 验证保存元数据默认设置
import re

with open('image_directory_nodes.py', 'r', encoding='utf-8') as f:
    content = f.read()

# 查找所有保存元数据的设置
pattern = r'"保存元数据".*?"default":\s*(True|False)'
matches = re.findall(pattern, content)

print(f"找到 {len(matches)} 个保存元数据设置:")
for i, match in enumerate(matches, 1):
    status = "✓ 开启" if match == "True" else "✗ 关闭"
    print(f"  {i}. {status}")

if all(m == "True" for m in matches):
    print("\n✅ 所有保存元数据设置已默认开启")
else:
    print("\n❌ 部分保存元数据设置未开启")
