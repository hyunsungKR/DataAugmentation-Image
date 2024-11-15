import os
import cv2
import numpy as np
import random
from albumentations import (
    Compose, RandomBrightnessContrast, HueSaturationValue, RandomGamma,
    CLAHE, RGBShift, Blur, MotionBlur, MedianBlur, GaussNoise, ImageCompression,
    Sharpen, Emboss, HorizontalFlip, VerticalFlip, OneOf
)
from PIL import Image

# 경로 설정
base_dir = r"path"  # 원본 이미지와 라벨 파일이 있는 기본 폴더 경로 설정함
target_total_count = 6987  # 목표 총 이미지 개수 설정 (원본 + 증강 이미지 포함)

augmentation_folder_suffix = "_aug"  # 증강된 이미지가 저장될 폴더의 접미사를 정의함

# 증강 파이프라인 설정 (밝기, 대비, 선명도 등 조정하여 여러 환경에 대응 가능하게 만듦)
augmentation_pipeline = Compose([
    RandomBrightnessContrast(brightness_limit=(0.0, 0.15), contrast_limit=(0.0, 0.15), p=0.5), # 밝기와 대비를 조정해 다양한 조명 조건에서도 모델이 강건성을 갖추도록 함
    # HueSaturationValue(hue_shift_limit=10, sat_shift_limit=20, val_shift_limit=10, p=0.5), # 색조, 채도, 밝기를 조정해 다양한 색상 조건을 학습시킴
    # RandomGamma(gamma_limit=(80, 120), p=0.5), # 감마 조절로 명암을 조정해 이미지의 다이나믹 레인지에 대응할 수 있도록 함
    CLAHE(clip_limit=4.0, tile_grid_size=(8, 8), p=0.5), # CLAHE를 적용해 이미지의 명암을 높임으로써 어두운 부분의 디테일을 강조함
    # RGBShift(r_shift_limit=10, g_shift_limit=10, b_shift_limit=10, p=0.5), # RGB 색상 값을 조정해 색상 환경 변화에 대응하도록 학습시킴
    # OneOf([
    #     Blur(blur_limit=3, p=0.3),
    #     MedianBlur(blur_limit=3, p=0.3),
    #     MotionBlur(blur_limit=3, p=0.3),
    # ], p=0.3), # 다양한 블러 효과를 적용하여 이미지가 흐려지는 조건을 학습할 수 있도록 함
    # GaussNoise(var_limit=(10.0, 50.0), p=0.3), # 가우스 노이즈를 추가하여 촬영 환경에서 발생할 수 있는 잡음에도 대응할 수 있게 함
    # Sharpen(alpha=(0.2, 0.5), lightness=(0.5, 1.0), p=0.3), # 이미지를 선명하게 만들어 중요한 특징이 강조되도록 함
    # Emboss(alpha=(0.2, 0.5), strength=(0.5, 1.0), p=0.3), # 엠보싱 효과를 주어 텍스처와 윤곽을 더 강조함
    # ImageCompression(quality_lower=70, quality_upper=100, p=0.5), # 이미지 압축을 적용해 낮은 품질의 이미지에서도 모델이 학습할 수 있도록 함
    HorizontalFlip(p=0.5), # 좌우 반전을 적용해 모델이 좌우 방향의 변화에도 강건성을 갖추도록 함
    VerticalFlip(p=0.5) # 상하 반전을 적용해 상하 위치 변화에도 적응할 수 있도록 함
], bbox_params={'format': 'yolo', 'label_fields': ['class_labels'], 'min_visibility': 0.1})

# 바운딩박스 로드 함수
def load_bounding_boxes(txt_path):
    with open(txt_path, "r") as f:
        lines = f.readlines()
    return [line.strip().split() for line in lines]

# 바운딩박스 저장 함수
def save_bounding_boxes(txt_path, bboxes):
    with open(txt_path, "w") as f:
        for bbox in bboxes:
            f.write(" ".join(map(str, bbox)) + "\n")

# 바운딩박스 좌표 클리핑 함수
def clip_bbox(bbox):
    x_center, y_center, width, height = bbox
    x_min = x_center - width / 2
    y_min = y_center - height / 2
    x_max = x_center + width / 2
    y_max = y_center + height / 2

    x_min = min(max(x_min, 0.0), 1.0)
    y_min = min(max(y_min, 0.0), 1.0)
    x_max = min(max(x_max, 0.0), 1.0)
    y_max = min(max(y_max, 0.0), 1.0)

    clipped_x_center = (x_min + x_max) / 2
    clipped_y_center = (y_min + y_max) / 2
    clipped_width = x_max - x_min
    clipped_height = y_max - y_min

    return [clipped_x_center, clipped_y_center, clipped_width, clipped_height]

# 증강 수행 함수
def augment_image_and_bboxes(image, bboxes, class_labels):
    augmented = augmentation_pipeline(image=image, bboxes=bboxes, class_labels=class_labels)
    return augmented['image'], augmented['bboxes'], class_labels

# 파일 단위 랜덤 증강
def augment_dataset():
    # 모든 파일 경로 수집
    all_files = []
    for root, dirs, files in os.walk(base_dir):
        images = [file for file in files if file.endswith(".jpg")]
        txt_files = [file for file in files if file.endswith(".txt")]
        for img in images:
            txt = img.replace(".jpg", ".txt")
            if txt in txt_files:
                all_files.append((os.path.join(root, img), os.path.join(root, txt)))

    original_count = len(all_files)
    required_augmented_images = target_total_count - original_count

    if required_augmented_images <= 0:
        print(f"증강이 필요하지 않습니다. 원본 데이터가 이미 목표 수({target_total_count}개)에 도달했습니다.")
        return

    print(f"원본 이미지 수: {original_count}, 목표 총 이미지 수: {target_total_count}, 증강 필요 수: {required_augmented_images}")

    # 파일 리스트 랜덤 셔플
    random.shuffle(all_files)
    data_count = 0

    while data_count < required_augmented_images:
        if not all_files:
            print("모든 파일이 증강되었지만 목표에 도달하지 못했습니다.")
            break

        img_path, txt_path = all_files.pop(0)  # 랜덤으로 선택된 파일
        root_dir = os.path.dirname(img_path)
        aug_dir = f"{root_dir}{augmentation_folder_suffix}"
        os.makedirs(aug_dir, exist_ok=True)

        image = cv2.imread(img_path)
        if image is None:
            print(f"이미지 로드 실패: {img_path}")
            continue
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        bboxes_raw = load_bounding_boxes(txt_path)
        yolo_bboxes = [[float(b[1]), float(b[2]), float(b[3]), float(b[4])] for b in bboxes_raw]
        class_labels = [b[0] for b in bboxes_raw]

        try:
            aug_image, aug_bboxes, class_labels = augment_image_and_bboxes(image, yolo_bboxes, class_labels)
            aug_img_name = os.path.basename(img_path).replace(".jpg", f"_aug_{data_count}.jpg")
            aug_txt_name = os.path.basename(txt_path).replace(".txt", f"_aug_{data_count}.txt")
            aug_img_path = os.path.join(aug_dir, aug_img_name)
            aug_txt_path = os.path.join(aug_dir, aug_txt_name)

            aug_image_bgr = cv2.cvtColor(aug_image, cv2.COLOR_RGB2BGR)
            cv2.imwrite(aug_img_path, aug_image_bgr)
            aug_bboxes_norm = [[class_labels[i]] + list(bbox) for i, bbox in enumerate(aug_bboxes)]
            save_bounding_boxes(aug_txt_path, aug_bboxes_norm)

            data_count += 1
            print(f"증강 완료: {aug_img_path}, 총 증강 수: {data_count}/{required_augmented_images}")
        except Exception as e:
            print(f"증강 실패: {img_path}, 에러: {e}")

    print(f"증강 작업 완료. 총 이미지 수: {target_total_count}에 도달.")

# 실행
augment_dataset()