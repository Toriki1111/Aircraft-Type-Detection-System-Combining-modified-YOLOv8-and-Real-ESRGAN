import pandas as pd

df = pd.read_csv("dataset/val.csv")

# Đếm số lượng mẫu của từng Class
class_counts = df["Labels"].value_counts().sort_index()

print("📊 PHÂN BỐ DỮ LIỆU TỪNG CLASS TRONG VAL.CSV:")
print("=" * 45)
for class_id, count in class_counts.items():
    print(f"Class {class_id:2d}: {count:4d} ảnh")
print("=" * 45)

# Kiểm tra độ lệch
max_count = class_counts.max()
min_count = class_counts.min()
print(f"📈 Class nhiều nhất: {max_count} ảnh")
print(f"📉 Class ít nhất:    {min_count} ảnh")

if max_count / min_count > 2:
    print("⚠️ Dữ liệu đang bị BẤT CÂN BẰNG (Imbalanced)!")
else:
    print("✅ Dữ liệu tương đối CÂN BẰNG (Balanced)!")