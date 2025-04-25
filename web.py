#!/usr/bin/env python3
# -*- coding: utf-8 -*-

# ------------------------------ 导入所需模块 ------------------------------
import os
import shutil
import random
import re
from collections import defaultdict
from flask import Flask, request, render_template_string, send_from_directory, redirect, url_for
from ultralytics import YOLO
from PIL import Image, ImageDraw, ImageFont
import cv2
import numpy as np
import csv
import torch
import torch.nn as nn
from torchvision import transforms, models
from torch.utils.data import Dataset, DataLoader
import pandas as pd

# ------------------------------ 定义映射字典与字体 ------------------------------
composition_mapping = {
    0: "cystic/spongiform",
    1: "mixed cystic & solid",
    2: "solid"
}

echogenicity_mapping = {
    0: "anechoic",
    1: "hyper-/isoechoic",
    2: "hypoechoic",
    3: "very hypoechoic"
}

shape_mapping = {
    0: "wider than tall",
    3: "taller than wide"
}

margin_mapping = {
    0: "smooth/ill-defined",
    2: "lobulated/irregular",
    3: "extra-thyroidal extension"
}

foci_mapping = {
    0: "none/large comet-tail artifact",
    1: "macrocalcifications",
    2: "peripheral/rim calcifications",
    3: "punctate echogenic foci"
}

try:
    font = ImageFont.truetype("DejaVuSans-Bold.ttf", 12)
except Exception as e:
    print("Error loading font, using default font. Error:", e)
    font = ImageFont.load_default()

# ------------------------------ 配置全局变量 ------------------------------
UPLOAD_FOLDER = 'uploads'
ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg', 'gif'}

# YOLO检测结果保存目录（确保当前用户有写入权限）
YOLO_RESULT_DIR = '/home/zihan/project1220/fold5_inference_detect_result'
# 裁剪后图片的保存路径（用于分类预测），但网页展示图片直接来自 YOLO_RESULT_DIR
YOLO_DIR = os.path.join(YOLO_RESULT_DIR, 'cropped')
# 标注后图片的输出目录（本次最终展示的图片将从这里读取）
ANNOTATION_OUTPUT_DIR = os.path.join(YOLO_RESULT_DIR, "annotate")

# ------------------------------ 初始化 Flask 应用 ------------------------------
app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
if not os.path.exists(UPLOAD_FOLDER):
    os.makedirs(UPLOAD_FOLDER)

# ------------------------------ Flask 辅助函数 ------------------------------
def allowed_file(filename):
    """检查文件后缀是否允许上传"""
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

# ------------------------------ YOLO 检测与裁剪处理 ------------------------------
def run_yolo_prediction():
    """
    使用 YOLO 对 UPLOAD_FOLDER 中的图片进行检测，
    将检测结果（包括 label 文件）保存到 YOLO_RESULT_DIR 中。
    如果检测过程中生成了子文件夹 'fold5_inference_detect_result'，则将其所有内容移动到 YOLO_RESULT_DIR 下，
    并删除该子文件夹；同时过滤掉没有检测到目标的图片。
    """
    path = UPLOAD_FOLDER  # 使用上传图片的文件夹
    model = YOLO(r'/home/zihan/project1220/fold_5_box/weights/best.pt')  # 修改为实际模型路径
    # 执行预测，保存结果到 YOLO_RESULT_DIR/fold5_inference_detect_result
    results = model.predict(
        source=path,
        max_det=1,
        conf=0.6,
        save=True,
        save_txt=True,
        project=YOLO_RESULT_DIR,
        name='fold5_inference_detect_result'
    )
    
    # 处理生成的嵌套文件夹问题
    result_dir_temp = os.path.join(YOLO_RESULT_DIR, "fold5_inference_detect_result")
    if os.path.exists(result_dir_temp):
        for item in os.listdir(result_dir_temp):
            src_path = os.path.join(result_dir_temp, item)
            dst_path = os.path.join(YOLO_RESULT_DIR, item)
            if os.path.exists(dst_path):
                if os.path.isdir(dst_path):
                    shutil.rmtree(dst_path)
                else:
                    os.remove(dst_path)
            shutil.move(src_path, YOLO_RESULT_DIR)
        shutil.rmtree(result_dir_temp)
    
    # 检测结果直接保存在 YOLO_RESULT_DIR 下
    result_dir = YOLO_RESULT_DIR
    labels_dir = os.path.join(result_dir, 'labels')
    
    # 过滤掉没有检测到目标的图像（对应 label 文件不存在或为空）
    image_extensions = ('.jpg', '.jpeg', '.png')
    image_files = [f for f in os.listdir(result_dir) if f.lower().endswith(image_extensions)]
    for image_file in image_files:
        base_name = os.path.splitext(image_file)[0]
        label_file = base_name + '.txt'
        label_path = os.path.join(labels_dir, label_file)
        if not os.path.exists(label_path) or os.path.getsize(label_path) == 0:
            print(f"Removing image {image_file} because no detection found.")
            os.remove(os.path.join(result_dir, image_file))
    return result_dir, labels_dir

def resize_if_needed(roi, target_size=224):
    """如果 ROI 超过目标尺寸，则按比例缩放到目标尺寸以内"""
    h, w = roi.shape[:2]
    if h > target_size or w > target_size:
        scale = target_size / max(h, w)
        new_h, new_w = int(h * scale), int(w * scale)
        roi = cv2.resize(roi, (new_w, new_h), interpolation=cv2.INTER_AREA)
    return roi

def center_crop_and_pad(roi, target_size=224):
    """将 ROI 居中放置于指定大小的黑色背景上"""
    h, w = roi.shape[:2]
    square_patch = np.zeros((target_size, target_size, 3), dtype=np.uint8)
    start_x = (target_size - w) // 2
    start_y = (target_size - h) // 2
    square_patch[start_y:start_y+h, start_x:start_x+w] = roi
    return square_patch

def crop_and_process_image(image_path, label_path, output_dir, target_size=224):
    """
    读取 label 文件中的检测框，对 image_path 指定的图片进行裁剪，
    对裁剪区域进行 resize 及居中填充后保存到 output_dir 中。
    """
    image = cv2.imread(image_path)
    if image is None:
        print(f"Error reading image {image_path}")
        return
    img_h, img_w = image.shape[:2]
    with open(label_path, 'r') as f:
        lines = f.readlines()
    if not lines:
        print(f"No labels found in {label_path}, skipping image.")
        return
    for i, line in enumerate(lines):
        tokens = line.strip().split()
        if len(tokens) < 5:
            print(f"Invalid label in {label_path}: {line}")
            continue
        try:
            cls = tokens[0]
            x_center_rel = float(tokens[1])
            y_center_rel = float(tokens[2])
            width_rel = float(tokens[3])
            height_rel = float(tokens[4])
        except Exception as e:
            print(f"Error parsing label in {label_path}: {line}, {e}")
            continue
        x_center = x_center_rel * img_w
        y_center = y_center_rel * img_h
        box_width = width_rel * img_w
        box_height = height_rel * img_h
        x_min = int(round(x_center - box_width / 2))
        y_min = int(round(y_center - box_height / 2))
        x_max = int(round(x_center + box_width / 2))
        y_max = int(round(y_center + box_height / 2))
        x_min = max(x_min, 0)
        y_min = max(y_min, 0)
        x_max = min(x_max, img_w)
        y_max = min(y_max, img_h)
        roi = image[y_min:y_max, x_min:x_max]
        print(f"\nProcessing image: {os.path.basename(image_path)} - Detection {i+1}:")
        print(f"Original bbox: ({x_min}, {y_min}), ({x_max}, {y_max}), size: {x_max-x_min} x {y_max-y_min}")
        roi = resize_if_needed(roi, target_size)
        new_h, new_w = roi.shape[:2]
        print(f"After resize: {new_w} x {new_h}")
        final_image = center_crop_and_pad(roi, target_size)
        print(f"Final image size: {final_image.shape[1]} x {final_image.shape[0]}")
        base_name = os.path.splitext(os.path.basename(image_path))[0]
        output_path = os.path.join(output_dir, f"{base_name}.png")
        cv2.imwrite(output_path, final_image)
        print(f"Saved cropped image to {output_path}")

def process_cropping(result_dir, labels_dir, output_dir, target_size=224):
    """
    遍历 result_dir 中的图片，根据对应 label 文件进行裁剪、resize 和居中填充，
    将处理后的图像保存到 output_dir 中。
    """
    os.makedirs(output_dir, exist_ok=True)
    image_extensions = ('.jpg', '.jpeg', '.png')
    image_files = [f for f in os.listdir(result_dir) if f.lower().endswith(image_extensions)]
    for image_file in image_files:
        image_path = os.path.join(result_dir, image_file)
        label_file = os.path.splitext(image_file)[0] + '.txt'
        label_path = os.path.join(labels_dir, label_file)
        if os.path.exists(label_path) and os.path.getsize(label_path) > 0:
            crop_and_process_image(image_path, label_path, output_dir, target_size)
        else:
            print(f"Label file not found or empty for {image_file}, skipping cropping.")

# ------------------------------ 分类预测与结果聚合 ------------------------------
class CustomDataset(Dataset):
    """
    从指定文件夹中读取图片（裁剪后的图像），不依赖 CSV 文件。
    """
    def __init__(self, image_folder, transform=None):
        self.samples = []
        self.transform = transform
        for entry in os.listdir(image_folder):
            if entry.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp', '.gif')):
                full_path = os.path.join(image_folder, entry)
                self.samples.append(full_path)
        self.samples.sort()
    def __len__(self):
        return len(self.samples)
    def __getitem__(self, idx):
        image_path = self.samples[idx]
        image = Image.open(image_path).convert('RGB')
        if self.transform:
            image = self.transform(image)
        return image, image_path

class MultiTaskResNet(nn.Module):
    """
    基于预训练 ResNet101 的多任务模型，
    分别输出 composition、echogenicity、shape、margin、foci 共 5 个任务的预测结果。
    """
    def __init__(self):
        super().__init__()
        backbone = models.resnet101(pretrained=True)
        num_ftrs = backbone.fc.in_features
        backbone.fc = nn.Identity()
        self.backbone = backbone
        self.dropout = nn.Dropout(p=0.3)
        self.head_composition = nn.Linear(num_ftrs, 3)
        self.head_echogenicity = nn.Linear(num_ftrs, 4)
        self.head_shape = nn.Linear(num_ftrs, 2)
        self.head_margin = nn.Linear(num_ftrs, 3)
        self.head_foci = nn.Linear(num_ftrs, 4)
    def forward(self, x):
        feat = self.backbone(x)
        feat = self.dropout(feat)
        logits_comp = self.head_composition(feat)
        logits_echo = self.head_echogenicity(feat)
        logits_shape = self.head_shape(feat)
        logits_margin = self.head_margin(feat)
        logits_foci = self.head_foci(feat)
        return logits_comp, logits_echo, logits_shape, logits_margin, logits_foci

def save_predictions_to_csv(model, dataloader, device, output_csv="predictions_v101_v1.csv"):
    """
    对测试集图片进行预测，并将每张图片的预测结果保存到 CSV 文件中；
    对 shape 与 margin 的预测结果进行逆映射，保证格式与原始设计一致。
    """
    model.eval()
    rows = []
    inverse_shape_map = {0: 0, 1: 3}
    inverse_margin_map = {0: 0, 1: 2, 2: 3}
    with torch.no_grad():
        for images, image_paths in dataloader:
            images = images.to(device)
            logits_comp, logits_echo, logits_shape, logits_margin, logits_foci = model(images)
            preds_comp = torch.argmax(logits_comp, dim=1).cpu().numpy()
            preds_echo = torch.argmax(logits_echo, dim=1).cpu().numpy()
            preds_shape_mapped = torch.argmax(logits_shape, dim=1).cpu().numpy()
            preds_margin_mapped = torch.argmax(logits_margin, dim=1).cpu().numpy()
            preds_foci = torch.argmax(logits_foci, dim=1).cpu().numpy()
            preds_shape_original = [inverse_shape_map.get(m, -1) for m in preds_shape_mapped]
            preds_margin_original = [inverse_margin_map.get(m, -1) for m in preds_margin_mapped]
            for i in range(len(image_paths)):
                row = [
                    image_paths[i],
                    preds_comp[i],
                    preds_echo[i],
                    preds_shape_original[i],
                    preds_margin_original[i],
                    preds_foci[i]
                ]
                rows.append(row)
    header = [
        "image_path",
        "composition_pred",
        "echogenicity_pred",
        "shape_pred(original)",
        "margin_pred(original)",
        "foci_pred"
    ]
    with open(output_csv, mode="w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(header)
        writer.writerows(rows)
    print(f"Predictions saved to {output_csv}")

def extract_nodule_id(image_path):
    """从文件名 image_{nodule_id}_{frame}.png 中提取 nodule_id"""
    match = re.search(r'image_(\d+)_', image_path)
    return match.group(1) if match else None

def extract_frame(image_path):
    """从文件名 image_{nodule_id}_{frame}.png 中提取 frame 数字"""
    match = re.search(r'image_\d+_(\d+)\.png', image_path)
    return int(match.group(1)) if match else None

def majority_vote(series):
    """对一个 pandas Series 进行众数统计，若存在多个众数则返回最小值"""
    return series.mode().iloc[0]

def points_to_tirads(points):
    """
    根据总分转换为 TI-RADS 分级：
      TR1: 0分
      TR2: 2分
      TR3: 3分
      TR4: 4-6分
      TR5: ≥7分
    """
    if points == 0:
        return 1
    elif points == 2:
        return 2
    elif points == 3:
        return 3
    elif 4 <= points <= 6:
        return 4
    elif points >= 7:
        return 5
    else:
        return None

def aggregate_predictions(input_csv="predictions_v101_v1.csv", output_csv="nodule_tirads_predictions_v1.csv"):
    """
    读取预测结果 CSV 文件，根据图片文件名提取 nodule_id，
    将同一 nodule 的中间 50% 帧数据进行众数统计，计算总预测分并转换为 TI-RADS 分级，
    将聚合结果保存到 CSV 文件中。
    """
    df = pd.read_csv(input_csv)
    df['nodule_id'] = df['image_path'].apply(extract_nodule_id)
    results = []
    grouped = df.groupby('nodule_id')
    for nodule_id, group in grouped:
        group = group.copy()
        group['frame'] = group['image_path'].apply(extract_frame)
        group = group.sort_values('frame')
        n = len(group)
        if n == 0:
            continue
        lower = int(n * 0.25)
        upper = int(n * 0.75)
        selected_group = group.iloc[lower:upper] if upper > lower else group
        composition_pred = majority_vote(selected_group['composition_pred'])
        echogenicity_pred = majority_vote(selected_group['echogenicity_pred'])
        shape_pred = majority_vote(selected_group['shape_pred(original)'])
        margin_pred = majority_vote(selected_group['margin_pred(original)'])
        foci_pred = majority_vote(selected_group['foci_pred'])
        total_pred_points = (composition_pred + echogenicity_pred + shape_pred + margin_pred + foci_pred)
        pred_tirads_level = points_to_tirads(total_pred_points)
        results.append({
            'nodule_id': nodule_id,
            'composition_pred': composition_pred,
            'echogenicity_pred': echogenicity_pred,
            'shape_pred': shape_pred,
            'margin_pred': margin_pred,
            'foci_pred': foci_pred,
            'total_pred_points': total_pred_points,
            'pred_tirads_level': pred_tirads_level
        })
    result_df = pd.DataFrame(results)
    result_df = result_df[['nodule_id',
                           'composition_pred',
                           'echogenicity_pred',
                           'shape_pred',
                           'margin_pred',
                           'foci_pred',
                           'total_pred_points',
                           'pred_tirads_level']]
    result_df.to_csv(output_csv, index=False)
    print(f"Aggregated nodule predictions saved to {output_csv}")

# ------------------------------ 标注相关函数 ------------------------------
def annotate_image(image_path, comp, echo, shape, margin, foci, output_path, margin_edge=5, line_spacing=2):
    """
    在 image_path 指定的图片右上角标注五个类别的文字信息，每个类别占一行，标注后保存到 output_path。
    """
    try:
        image = Image.open(image_path).convert("RGB")
    except Exception as e:
        print(f"Error opening image {image_path}: {e}")
        return

    draw = ImageDraw.Draw(image)
    text_lines = [
        f"composition: {composition_mapping.get(comp, str(comp))}",
        f"echogenicity: {echogenicity_mapping.get(echo, str(echo))}",
        f"shape: {shape_mapping.get(shape, str(shape))}",
        f"margin: {margin_mapping.get(margin, str(margin))}",
        f"foci: {foci_mapping.get(foci, str(foci))}",
    ]
    y = margin_edge
    for line in text_lines:
        bbox = draw.textbbox((0, 0), line, font=font)
        text_width = bbox[2] - bbox[0]
        text_height = bbox[3] - bbox[1]
        x = image.width - margin_edge - text_width
        draw.text((x, y), line, font=font, fill=(255, 0, 0))
        y += text_height + line_spacing
    try:
        image.save(output_path)
        print(f"Annotated image saved to {output_path}")
    except Exception as e:
        print(f"Error saving annotated image {output_path}: {e}")

def annotate_all():
    """
    根据 CSV 文件中的预测结果，对 INPUT_IMAGE_DIR 中对应的图片进行标注，
    并将标注后的图片保存到 ANNOTATION_OUTPUT_DIR 中。
    其中，CSV 文件要求包含如下列（不作修改）： 
      image_path, composition_pred, echogenicity_pred, shape_pred(original), margin_pred(original), foci_pred
    且本例中 INPUT_IMAGE_DIR 下的图片以 .jpg 结尾。
    """
    # 使用与您提供的annotation脚本中相同的路径定义
    INPUT_IMAGE_DIR = "/home/zihan/project1220/fold5_inference_detect_result"
    CSV_FILE = "/home/zihan/project1220/predictions_v101_v1.csv"
    if not os.path.exists(ANNOTATION_OUTPUT_DIR):
        os.makedirs(ANNOTATION_OUTPUT_DIR)
    try:
        df = pd.read_csv(CSV_FILE)
    except Exception as e:
        print("Error reading CSV file:", e)
        return
    for idx, row in df.iterrows():
        base_name, _ = os.path.splitext(os.path.basename(row["image_path"]))
        src_image_path = os.path.join(INPUT_IMAGE_DIR, base_name + ".jpg")
        if not os.path.exists(src_image_path):
            print(f"Image not found: {src_image_path}")
            continue
        try:
            comp = int(row["composition_pred"])
            echo = int(row["echogenicity_pred"])
            shape_val = int(row["shape_pred(original)"])
            margin_val = int(row["margin_pred(original)"])
            foci_val = int(row["foci_pred"])
        except Exception as e:
            print(f"Error parsing category values for {base_name}: {e}")
            continue
        out_image_path = os.path.join(ANNOTATION_OUTPUT_DIR, base_name + ".jpg")
        annotate_image(src_image_path, comp, echo, shape_val, margin_val, foci_val, out_image_path)

def process_all():
    """
    综合处理流程：
      1. 使用 YOLO 对 UPLOAD_FOLDER 中的图片进行检测，并将结果保存到 YOLO_RESULT_DIR；
      2. 根据检测结果，对图片进行裁剪、resize 和居中填充（保存到 YOLO_RESULT_DIR/cropped，用于分类预测）；
      3. 对裁剪后的图像进行分类预测并聚合计算 TI-RADS 分级，将预测结果保存到 CSV 文件中；
      4. 调用标注函数，对检测图片进行标注，标注结果保存到 ANNOTATION_OUTPUT_DIR 下；
      5. 返回用于网页展示的图片列表（优先展示标注后的图片）及聚合后的 CSV 数据。
    """
    print("Starting YOLO detection ...")
    result_dir, labels_dir = run_yolo_prediction()
    
    # 裁剪后图片（用于分类预测）保存到 YOLO_RESULT_DIR/cropped
    cropped_dir = os.path.join(result_dir, 'cropped')
    process_cropping(result_dir, labels_dir, cropped_dir, target_size=224)
    print("YOLO detection and cropping completed.")
    
    print("Starting classification prediction and aggregation...")
    transform_test = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])
    test_dataset = CustomDataset(cropped_dir, transform=transform_test)
    test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False, num_workers=4)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = MultiTaskResNet().to(device)
    model_path = "/home/zihan/project1220/multi_task_resnet101_fold1_v1.pt"  # 修改为实际模型路径
    state_dict = torch.load(model_path, map_location=device)
    model.load_state_dict(state_dict)
    save_predictions_to_csv(model, test_loader, device, output_csv="predictions_v101_v1.csv")
    aggregate_predictions(input_csv="predictions_v101_v1.csv", output_csv="nodule_tirads_predictions_v1.csv")
    print("Classification prediction and aggregation completed.")
    
    # 调用标注函数，对检测图片进行标注，结果保存到 ANNOTATION_OUTPUT_DIR 下
    annotate_all()
    
    # 读取聚合后的预测数据
    if os.path.exists("nodule_tirads_predictions_v1.csv"):
        agg_df = pd.read_csv("nodule_tirads_predictions_v1.csv")
        aggregated_data = agg_df.to_dict(orient="records")
    else:
        aggregated_data = []
    
    # 列举用于网页展示的图片——优先读取标注后的图片
    detection_images = []
    if os.path.exists(ANNOTATION_OUTPUT_DIR):
        print("os.path.exists(ANNOTATION_OUTPUT_DIR)")
        folder_to_list = ANNOTATION_OUTPUT_DIR
    else:
        folder_to_list = result_dir
        print("ANNOTATION_OUTPUT_DIR 不存在，使用 result_dir")
    
    for fname in os.listdir(folder_to_list):
        fpath = os.path.join(folder_to_list, fname)
        if os.path.isfile(fpath) and fname.lower().split('.')[-1] in ALLOWED_EXTENSIONS:
            # 使用正则表达式解析文件名，匹配例如 "image_12_34.jpg"
            match = re.match(r'image_(\d+)_(\d+)\.', fname)
            if match:
                i_num = int(match.group(1))
                j_num = int(match.group(2))
            else:
                # 如果不匹配则默认设为 0
                i_num = 0
                j_num = 0
            detection_images.append({
                'filename': fname,
                'i': i_num,
                'j': j_num
            })
    
    # 按照 j 值进行升序排序
    detection_images = sorted(detection_images, key=lambda x: x['j'])
    
    return detection_images, aggregated_data

INDEX_HTML = """
<!DOCTYPE html>
<html lang="en">
<head>
  <meta charset="UTF-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <title>Ultrasound Nodule Detection & TI-RADS Classification</title>
  <style>
    /* 全局样式重置 */
    * { margin: 0; padding: 0; box-sizing: border-box; }
    html, body {
      height: 100%;
      font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, Helvetica, sans-serif;
    }
    /* 整体背景采用深灰色，视觉效果更柔和 */
    body {
      background: #1f1f1f;
      color: #fff;
      line-height: 1.6;
      overflow-x: hidden;
    }
    .container {
      width: 100%;
      overflow: hidden;
    }
    /* Hero区域 */
    .hero {
      min-height: 100vh;
      display: flex;
      flex-direction: column;
      justify-content: center;
      align-items: center;
      text-align: center;
      padding: 2rem 1rem;
    }
    .hero h1 {
      font-size: 3rem;
      font-weight: 600;
      margin-bottom: 1rem;
      color: #fff;
    }
    .hero p {
      font-size: 1.25rem;
      margin-bottom: 1.5rem;
      color: #fff;
    }
    /* 拖拽上传区域 */
    .drop-zone {
      border: 3px dashed #007AFF;  /* 使用蓝色边框 */
      border-radius: 12px;
      background-color: #2a2a2a;
      padding: 10rem;  /* 增大边框内部空间 */
      text-align: center;
      transition: background-color 0.3s ease;
      margin-bottom: 1rem;
      cursor: pointer;
      width: 100%;
      max-width: 1200px;  /* 提升上传区域的最大宽度 */
    }
    .drop-zone.dragover {
      background-color: #333;
    }
    .drop-zone__prompt {
      font-size: 1.5rem;
      margin-bottom: 0.5rem;
      color: #fff;
    }
    .supported {
      font-size: 1rem;
      color: #aaa;
      display: block;
      margin-top: 0.5rem;
    }
    .file-count {
      font-size: 1rem;
      color: #fff;
      display: block;
      margin-top: 1rem;
    }
    input[type="file"] {
      display: none;
    }
    .upload-btn {
      background-color: #007AFF;
      color: #fff;
      border: none;
      padding: 14px 28px;
      font-size: 1.25rem;
      border-radius: 8px;
      cursor: pointer;
      transition: background 0.3s ease;
      margin-top: 1rem;
    }
    .upload-btn:hover {
      background-color: #005bb5;
    }
    .nodule-note {
      margin-top: 1rem;
      font-size: 1rem;
      color: #fff;
    }
    /* Features展示区域 */
    .features {
      background-color: #242424;
      padding: 4rem 1rem;
      text-align: center;
    }
    .features h2 {
      font-size: 2.5rem;
      margin-bottom: 2rem;
      color: #fff;
      font-weight: 600;
    }
    /* 使用CSS Grid布局，两列排布，留白适中 */
    .features-grid {
      display: grid;
      grid-template-columns: repeat(2, 1fr);
      gap: 2rem;
      max-width: 1200px;
      margin: 0 auto;
    }
    .feature-block {
      background-color: #444;
      border-radius: 16px;
      padding: 2rem;
      color: #fff;
      display: flex;
      flex-direction: column;
      justify-content: center;
      text-align: left;
      min-height: 180px;
      transition: transform 0.3s ease, background-color 0.3s ease;
    }
    .feature-block:hover {
      transform: translateY(-5px);
      background-color: #555;
    }
    .feature-block h3 {
      font-size: 1.75rem;
      margin-bottom: 1rem;
      font-weight: 600;
      color: #fff;
    }
    .feature-block p {
      font-size: 1.125rem;
      line-height: 1.5;
      color: #fff;
    }
    /* 自适应：小屏幕下单列排布 */
    @media (max-width: 768px) {
      .features-grid {
        grid-template-columns: 1fr;
      }
      .hero h1 {
        font-size: 2.5rem;
      }
      .hero p {
        font-size: 1.125rem;
      }
    }
  </style>
</head>
<body>
  <div class="container">
    <!-- Hero上传区域 -->
    <div class="hero">
      <h1>Ultrasound Nodule Detection & TI-RADS Classification</h1>
      <p>Experience the power of modern AI in detecting and classifying ultrasound images seamlessly.</p>
      <form method="POST" enctype="multipart/form-data">
        <!-- 支持拖拽及点击上传 -->
        <div class="drop-zone" id="drop-zone">
          <span class="drop-zone__prompt">Drag or drop files here or click to select images</span>
          <span class="supported">Supported formats: PNG, JPG, JPEG, GIF</span>
          <span id="file-count" class="file-count"></span>
          <input type="file" id="file-upload" name="files" accept="image/png, image/jpg, image/jpeg, image/gif" multiple required class="drop-zone__input">
        </div>
        <button type="submit" class="upload-btn">Upload and Analyze</button>
        <div class="nodule-note">Note: Uploaded files must belong to the same nodule.</div>
      </form>
    </div>
    <!-- Features展示区域 -->
    <div class="features">
      <h2>Our Capabilities</h2>
      <div class="features-grid">
        <div class="feature-block">
          <h3>Accurate Nodule Detection</h3>
          <p>Leverage advanced algorithms to precisely identify nodules from ultrasound imagery.</p>
        </div>
        <div class="feature-block">
          <h3>Automated Preprocessing</h3>
          <p>Experience seamless image cropping, resizing, and enhancement for optimal results.</p>
        </div>
        <div class="feature-block">
          <h3>Deep Learning Classification</h3>
          <p>Employ powerful neural networks to reliably compute TI-RADS scores and insights.</p>
        </div>
        <div class="feature-block">
          <h3>Smart Annotation</h3>
          <p>Overlay detailed labels on images, highlighting key features with precision.</p>
        </div>
        <div class="feature-block">
          <h3>Interactive Results Viewer</h3>
          <p>Engage with real-time analysis through interactive displays built for clarity.</p>
        </div>
      </div>
    </div>
  </div>
  <script>
    // 获取元素
    const dropZone = document.getElementById('drop-zone');
    const fileInput = document.getElementById('file-upload');
    const fileCountElement = document.getElementById('file-count');
    
    // 点击触发文件选择
    dropZone.addEventListener('click', () => {
      fileInput.click();
    });
    
    // 拖拽中的效果
    dropZone.addEventListener('dragover', (e) => {
      e.preventDefault();
      dropZone.classList.add('dragover');
    });
    dropZone.addEventListener('dragleave', () => {
      dropZone.classList.remove('dragover');
    });
    
    // 拖拽后更新提示信息
    dropZone.addEventListener('drop', (e) => {
      e.preventDefault();
      dropZone.classList.remove('dragover');
      if (e.dataTransfer.files.length) {
        fileInput.files = e.dataTransfer.files;
        const count = e.dataTransfer.files.length;
        fileCountElement.textContent = count + " file(s) ready for upload";
      }
    });
    
    // 文件选择变化时也更新提示信息
    fileInput.addEventListener('change', () => {
      const count = fileInput.files.length;
      fileCountElement.textContent = count + " file(s) ready for upload";
    });
  </script>
</body>
</html>
"""

PROCESSING_HTML = """
<!DOCTYPE html>
<html lang="en">
<head>
  <meta charset="UTF-8">
  <meta http-equiv="refresh" content="3;url={{ url_for('results') }}">
  <meta name="viewport" content="width=device-width, initial-scale=1.0">
  <title>Processing Results</title>
  <style>
    body { 
      margin: 0; 
      padding: 0; 
      background-color: #000; 
      font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif; 
      display: flex; 
      justify-content: center; 
      align-items: center; 
      height: 100vh;
      color: #fff;
      text-align: center;
    }
    .container {
      background: #111; 
      padding: 30px; 
      border-radius: 8px; 
      box-shadow: 0 4px 10px rgba(0,0,0,0.5);
      max-width: 600px;
      width: 90%;
    }
    h1 {
      margin-bottom: 20px;
      color: #fff;
    }
    p {
      color: #ccc;
      margin: 0;
      padding: 10px 0;
    }
    .spinner {
      margin: 20px auto;
      width: 50px;
      height: 50px;
      border: 5px solid #333;
      border-top: 5px solid #007AFF;
      border-radius: 50%;
      animation: spin 1s linear infinite;
    }
    @keyframes spin {
      0% { transform: rotate(0deg); }
      100% { transform: rotate(360deg); }
    }
  </style>
</head>
<body>
  <div class="container">
    <h1>Processing Results</h1>
    <div class="spinner"></div>
    <p>Your images are being analyzed using the YOLO detection model and classification.</p>
    <p>Please wait, you will be redirected shortly...</p>
  </div>
</body>
</html>
"""

RESULTS_HTML = """
<!DOCTYPE html>
<html lang="en">
<head>
  <meta charset="UTF-8">
  <meta name="viewport" content="width=device-width, initial-scale=1.0">
  <title>Medical Imaging Analysis Report</title>
  <style>
    /* 医疗级配色方案 */
    :root {
      --primary: #00BCD4;       /* 主色调 - 医疗蓝 */
      --secondary: #4CAF50;     /* 辅助色 - 验证绿 */
      --background: #0A1929;    /* 深蓝背景 */
      --surface: #1A2A3A;       /* 表面色 */
      --text-primary: #FFFFFF;
      --tr1: #4CAF50;          /* TR1 绿色 */
      --tr2: #8BC34A;          /* TR2 浅绿 */
      --tr3: #FFC107;          /* TR3 黄色 */
      --tr4: #FF9800;          /* TR4 橙色 */
      --tr5: #F44336;          /* TR5 红色 */
    }

    body {
      margin: 0;
      padding: 2rem;
      background: var(--background);
      font-family: 'Roboto', sans-serif;
      color: var(--text-primary);
      position: relative;
    }

    .return-link-outside {
      position: fixed;
      top: 10px;
      left: 10px;
      background: #fff;
      color: var(--primary);
      padding: 10px 20px;
      border-radius: 4px;
      text-decoration: none;
      font-weight: 500;
      font-size: 1.25rem;
      z-index: 1000;
    }
    .return-link-outside:hover {
      background: #f0f0f0;
    }

    .container {
      max-width: 1200px;
      margin: 0 auto;
      background: var(--surface);
      border-radius: 16px;
      padding: 2rem;
      box-shadow: 0 8px 24px rgba(0,0,0,0.3);
    }

    /* 头部区域 */
    .report-header {
      display: flex;
      justify-content: space-between;
      align-items: center;
      margin-bottom: 1rem;
      padding-bottom: 1rem;
      border-bottom: 2px solid rgba(0,188,212,0.3);
    }
    .header-left {
      display: flex;
      flex-direction: column;
    }
    .header-title {
      font-size: 2rem;
      color: var(--primary);
      margin: 0;
    }
    .subtitle {
      font-size: 1rem;
      color: rgba(255,255,255,0.8);
      margin: 0.2rem 0 0 0;
    }

    /* TI-RADS 评分展示 */
    .tirads-display {
      background: linear-gradient(135deg, var(--surface), #132235);
      padding: 1.5rem;
      border-radius: 12px;
      border: 1px solid var(--primary);
      min-width: 240px;
      text-align: center;
    }
    .tirads-level {
      font-size: 3rem;
      font-weight: 700;
      margin: 0;
      color: var(--primary);
    }
    .tirads-text {
      font-size: 1.2rem;
      color: rgba(255,255,255,0.8);
    }
    
    /* 导航标签 */
    .nav-tabs {
      display: flex;
      justify-content: center;
      margin: 1rem 0;
    }
    .nav-tab {
      padding: 0.5rem 1rem;
      margin: 0 0.5rem;
      cursor: pointer;
      border: 1px solid var(--primary);
      border-radius: 4px;
      color: var(--primary);
      transition: background 0.3s;
    }
    .nav-tab.active {
      background: var(--primary);
      color: #fff;
    }

    /* 图片页面相关 */
    #imagesPage {
      display: block;
    }
    #tablePage {
      display: none;
    }
    .main-visualization {
      position: relative;
      background: #000;
      border-radius: 12px;
      overflow: hidden;
      margin-bottom: 0.5rem;
      border: 2px solid rgba(0,188,212,0.2);
    }
    .main-image {
      width: 100%;
      height: 500px;
      object-fit: contain;
      cursor: pointer; /* 提示用户图片可点击 */
    }
    .nav-button {
      position: absolute;
      top: 50%;
      transform: translateY(-50%);
      background: rgba(0,188,212,0.8);
      border: none;
      color: white;
      padding: 1rem;
      border-radius: 50%;
      cursor: pointer;
      transition: all 0.3s;
      z-index: 10;
    }
    .nav-button:hover {
      background: var(--primary);
      box-shadow: 0 4px 12px rgba(0,188,212,0.3);
    }
    .nav-button.prev {
      left: 1rem;
    }
    .nav-button.next {
      right: 1rem;
    }
    .image-caption {
      text-align: center;
      font-size: 1.2rem;
      margin-bottom: 0.3rem;
      color: var(--primary);
    }
    .image-count {
      text-align: center;
      font-size: 1rem;
      margin-bottom: 1rem;
      color: rgba(255,255,255,0.8);
    }
    .thumbnail-carousel {
      display: flex;
      overflow-x: auto;
      gap: 1rem;
      padding: 1rem 0;
      margin-bottom: 2rem;
    }
    .thumbnail-item {
      flex: 0 0 160px;
      height: 120px;
      border-radius: 8px;
      overflow: hidden;
      cursor: pointer;
      transition: all 0.3s;
      position: relative;
      border: 2px solid transparent;
    }
    .thumbnail-item.active {
      border-color: var(--primary);
      box-shadow: 0 0 12px rgba(0,188,212,0.3);
    }
    .thumbnail-item:hover {
      transform: translateY(-3px);
    }
    .thumbnail-img {
      width: 100%;
      height: 100%;
      object-fit: cover;
    }

    /* 表格页面相关 */
    .clinical-table {
      width: 100%;
      border-collapse: collapse;
      background: rgba(255,255,255,0.03);
      border-radius: 12px;
      overflow: hidden;
    }
    .clinical-table th, .clinical-table td {
      padding: 1rem;
      text-align: left;
    }
    .clinical-table th {
      background: rgba(0,188,212,0.15);
      font-weight: 500;
    }
    .clinical-table td {
      border-bottom: 1px solid rgba(255,255,255,0.05);
    }
    .clinical-table tr:last-child td {
      border-bottom: none;
    }

    /* TR等级颜色标记 */
    .tr-badge {
      display: inline-block;
      padding: 0.3rem 0.8rem;
      border-radius: 20px;
      font-weight: 500;
    }
    .tr1 { background: var(--tr1); color: #000; }
    .tr2 { background: var(--tr2); color: #000; }
    .tr3 { background: var(--tr3); color: #000; }
    .tr4 { background: var(--tr4); color: #fff; }
    .tr5 { background: var(--tr5); color: #fff; }

    /* 模态窗口样式 */
    .modal {
      display: none;
      position: fixed;
      z-index: 9999;
      left: 0;
      top: 0;
      width: 100%;
      height: 100%;
      overflow: auto;
      background-color: rgba(0,0,0,0.8);
    }
    .modal-content {
      margin: 5% auto;
      display: block;
      max-width: 90%;
      max-height: 80%;
    }
    .close {
      position: absolute;
      top: 20px;
      right: 35px;
      color: #f1f1f1;
      font-size: 40px;
      font-weight: bold;
      cursor: pointer;
    }
    .close:hover,
    .close:focus {
      color: #bbb;
    }
  </style>
</head>
<body>
  <!-- Return to Index 按钮 -->
  <a class="return-link-outside" href="{{ url_for('reset_and_index') }}">Return to Upload Page</a>
  
  <div class="container">
    <header class="report-header">
      <div class="header-left">
        <h1 class="header-title">Ultrasound Analysis Report</h1>
        <p class="subtitle">AI-Powered Thyroid Nodule Assessment</p>
      </div>
      <div class="tirads-display">
        <div class="tirads-level tr{{ aggregated_data[0].pred_tirads_level if aggregated_data|length > 0 else '0' }}">
          TR{{ aggregated_data[0].pred_tirads_level if aggregated_data|length > 0 else 'N/A' }}
        </div>
        <p class="tirads-text">TI-RADS Classification</p>
      </div>
    </header>

    <!-- 导航标签 -->
    <div class="nav-tabs">
      <div id="imagesTab" class="nav-tab active" onclick="showImages()">Images</div>
      <div id="tableTab" class="nav-tab" onclick="showTable()">Table</div>
    </div>
    
    <!-- Images 页面 -->
    <div id="imagesPage">
      <!-- 主图区域 -->
      <section class="main-visualization">
        <button class="nav-button prev" onclick="prevImage()">&#9664;</button>
        <img class="main-image" id="mainImage" 
             src="{{ url_for('annotate_file', filename=results[0].filename) }}" 
             alt="Medical Visualization" onclick="openModal(currentIndex)">
        <button class="nav-button next" onclick="nextImage()">&#9654;</button>
      </section>
      
      <!-- 图片名称与数量展示 -->
      <div class="image-caption" id="imageCaption">
        {{ results[0].filename }}
      </div>
      <div class="image-count" id="imageCount">
        Image 1 / {{ results|length }}
      </div>
      
      <!-- 缩略图轮播 -->
      <div class="thumbnail-carousel">
        {% for result in results %}
        <div class="thumbnail-item {{ 'active' if loop.first }}" onclick="updateMainImageByIndex({{ loop.index0 }})">
          <img class="thumbnail-img" src="{{ url_for('annotate_file', filename=result.filename) }}" alt="Thumbnail {{ loop.index }}">
        </div>
        {% endfor %}
      </div>
    </div>
    
    <!-- Table 页面 -->
    <div id="tablePage">
      <table class="clinical-table">
        <thead>
          <tr>
            <th>Parameter</th>
            <th>Score</th>
            <th>Clinical Significance</th>
          </tr>
        </thead>
        <tbody>
          <tr>
            <td>Composition</td>
            <td>{{ aggregated_data[0].composition_pred if aggregated_data|length > 0 else 'N/A' }}</td>
            <td>Cystic vs Solid characteristics</td>
          </tr>
          <tr>
            <td>Echogenicity</td>
            <td>{{ aggregated_data[0].echogenicity_pred if aggregated_data|length > 0 else 'N/A' }}</td>
            <td>Tissue density comparison</td>
          </tr>
          <tr>
            <td>Margin</td>
            <td>{{ aggregated_data[0].margin_pred if aggregated_data|length > 0 else 'N/A' }}</td>
            <td>Boundary definition analysis</td>
          </tr>
          <tr>
            <td>Shape</td>
            <td>{{ aggregated_data[0].shape_pred if aggregated_data|length > 0 else 'N/A' }}</td>
            <td>Geometric appearance of the nodule</td>
          </tr>
          <tr>
            <td>Foci</td>
            <td>{{ aggregated_data[0].foci_pred if aggregated_data|length > 0 else 'N/A' }}</td>
            <td>Presence of calcifications or other bright spots</td>
          </tr>
          <tr>
            <td>Suspicion Level</td>
            <td>
              <span class="tr-badge tr{{ aggregated_data[0].pred_tirads_level if aggregated_data|length > 0 else '0' }}">
                TR{{ aggregated_data[0].pred_tirads_level if aggregated_data|length > 0 else 'N/A' }}
              </span>
            </td>
            <td>Recommended follow-up protocol</td>
          </tr>
        </tbody>
      </table>
    </div>
  </div>

  <!-- Modal for enlarged image -->
  <div id="myModal" class="modal">
    <span class="close" onclick="closeModal()">&times;</span>
    <img class="modal-content" id="modalImage">
    <div id="caption"></div>
  </div>

  <script>
    var imageList = {{ results|tojson }};
    var currentIndex = 0;
    var baseUrl = "{{ url_for('annotate_file', filename='') }}";

    function updateMainImageByIndex(index) {
      if (index < 0) {
        index = imageList.length - 1;
      } else if (index >= imageList.length) {
        index = 0;
      }
      currentIndex = index;
      var mainImg = document.getElementById('mainImage');
      var caption = document.getElementById('imageCaption');
      var imageCount = document.getElementById('imageCount');
      
      mainImg.src = baseUrl + imageList[index].filename;
      caption.textContent = imageList[index].filename;
      imageCount.textContent = "Image " + (index + 1) + " / " + imageList.length;
      
      var thumbnails = document.querySelectorAll('.thumbnail-item');
      thumbnails.forEach(function(item, idx) {
        item.classList.toggle('active', idx === index);
      });
    }
    
    function prevImage() {
      updateMainImageByIndex(currentIndex - 1);
    }
    
    function nextImage() {
      updateMainImageByIndex(currentIndex + 1);
    }
    
    function showImages() {
      document.getElementById("imagesPage").style.display = "block";
      document.getElementById("tablePage").style.display = "none";
      document.getElementById("imagesTab").classList.add("active");
      document.getElementById("tableTab").classList.remove("active");
    }
    
    function showTable() {
      document.getElementById("imagesPage").style.display = "none";
      document.getElementById("tablePage").style.display = "block";
      document.getElementById("tableTab").classList.add("active");
      document.getElementById("imagesTab").classList.remove("active");
    }
    
    function openModal(index) {
      currentIndex = index;
      document.getElementById("myModal").style.display = "block";
      document.getElementById("modalImage").src = baseUrl + imageList[currentIndex].filename;
    }
    
    function closeModal() {
      document.getElementById("myModal").style.display = "none";
    }
    
    window.onclick = function(event) {
      var modal = document.getElementById("myModal");
      if (event.target == modal) {
        modal.style.display = "none";
      }
    }
  </script>
</body>
</html>
"""
# ------------------------------ Flask 路由 ------------------------------

folder_to_list = ANNOTATION_OUTPUT_DIR
@app.route("/", methods=["GET", "POST"])
def index():
    """
    首页上传界面：
    - GET: 显示上传表单。
    - POST: 保存上传图片到 UPLOAD_FOLDER，并重定向到处理页面。
    """
    if request.method == "POST":
        files = request.files.getlist("files")
        for file in files:
            if file and allowed_file(file.filename):
                # 注意：实际使用中建议使用 secure_filename
                filename = file.filename  
                file.save(os.path.join(app.config['UPLOAD_FOLDER'], filename))
        return redirect(url_for("processing"))
    return render_template_string(INDEX_HTML)

@app.route("/processing")
def processing():
    """
    处理页面：显示提示信息，3秒后自动重定向到结果页面。
    """
    return render_template_string(PROCESSING_HTML)

@app.route("/results")
def results():
    """
    结果页面：
    - 执行综合处理流程，对上传图片进行检测、裁剪、分类预测与聚合；
    - 获取检测后图片（最终展示为标注后的图片）及聚合的 CSV 数据；
    - 显示检测图片及聚合数据表格。
    """
    detection_images, aggregated_data = process_all()
    return render_template_string(RESULTS_HTML, results=detection_images, aggregated_data=aggregated_data)

@app.route('/yolos/<filename>')
def yolos_file(filename):
    """
    返回存放在 YOLO_RESULT_DIR 中的检测结果图片（包括标注后的图片）。
    """
    return send_from_directory(YOLO_RESULT_DIR, filename)

@app.route('/annotate/<filename>')
def annotate_file(filename):
    return send_from_directory(ANNOTATION_OUTPUT_DIR, filename)

@app.route('/uploads/<filename>')
def uploaded_file(filename):
    """返回原始上传的文件"""
    return send_from_directory(app.config['UPLOAD_FOLDER'], filename)

@app.route("/reset")
def reset_and_index():
    """
    清空 UPLOAD_FOLDER 和 YOLO_RESULT_DIR（包括所有子文件夹），
    并重定向回上传页面。
    """
    # 清空上传目录
    folder = app.config['UPLOAD_FOLDER']
    for filename in os.listdir(folder):
        file_path = os.path.join(folder, filename)
        try:
            if os.path.isfile(file_path) or os.path.islink(file_path):
                os.unlink(file_path)
            elif os.path.isdir(file_path):
                shutil.rmtree(file_path)
        except Exception as e:
            print('Failed to delete %s. Reason: %s' % (file_path, e))
    # 清空检测结果目录
    if os.path.exists(YOLO_RESULT_DIR):
        shutil.rmtree(YOLO_RESULT_DIR)
    return redirect(url_for("index"))

if __name__ == "__main__":
    app.run(debug=True)