import os
import shutil
import random

source_dir = r"C:\Users\HP\OneDrive\الصور\سطح المكتب\car.ai\dataset"
train_dir  = r"C:\Users\HP\OneDrive\الصور\سطح المكتب\car.ai\data\train"
test_dir   = r"C:\Users\HP\OneDrive\الصور\سطح المكتب\car.ai\data\test"

split_ratio = 0.7

# امتدادات الصور فقط
valid_ext = ('.jpg', '.jpeg', '.png')

os.makedirs(train_dir, exist_ok=True)
os.makedirs(test_dir, exist_ok=True)

for class_name in os.listdir(source_dir):

    class_path = os.path.join(source_dir, class_name)

    if not os.path.isdir(class_path):
        continue

    # فلترة الصور فقط
    images = [img for img in os.listdir(class_path) if img.lower().endswith(valid_ext)]

    if len(images) == 0:
        print(f"⚠️ الفئة {class_name} فاضية")
        continue

    random.shuffle(images)

    split_index = int(len(images) * split_ratio)

    # 💡 ضمان وجود صور في الاثنين
    if split_index == 0:
        split_index = 1
    if split_index == len(images):
        split_index = len(images) - 1

    train_images = images[:split_index]
    test_images  = images[split_index:]

    os.makedirs(os.path.join(train_dir, class_name), exist_ok=True)
    os.makedirs(os.path.join(test_dir, class_name), exist_ok=True)

    for img in train_images:
        shutil.copy(os.path.join(class_path, img),
                    os.path.join(train_dir, class_name, img))

    for img in test_images:
        shutil.copy(os.path.join(class_path, img),
                    os.path.join(test_dir, class_name, img))

print("✅ تم التقسيم بدون فقدان أي فئة")