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
base_dir = "path"  # 원본 이미지와 라벨 파일이 있는 기본 폴더 경로 설정함
target_total_count = 3000  # 목표 총 이미지 개수 설정 (원본 + 증강 이미지 포함)

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

# 바운딩박스를 로드하는 함수
def load_bounding_boxes(txt_path):
    # 바운딩박스를 담은 텍스트 파일을 읽어옴
    with open(txt_path, "r") as f:
        lines = f.readlines()
    return [line.strip().split() for line in lines]

# 바운딩박스를 저장하는 함수
def save_bounding_boxes(txt_path, bboxes):
    # 바운딩박스 데이터를 YOLO 형식으로 텍스트 파일에 저장함
    with open(txt_path, "w") as f:
        for bbox in bboxes:
            f.write(" ".join(map(str, bbox)) + "\n")

# 바운딩박스 좌표가 [0.0, 1.0] 범위를 넘지 않도록 조정
def clip_bbox(bbox):
    x_center, y_center, width, height = bbox
    x_min = x_center - width / 2
    y_min = y_center - height / 2
    x_max = x_center + width / 2
    y_max = y_center + height / 2

    # x_min, y_min, x_max, y_max 값을 [0.0, 1.0] 범위 내로 조정함
    x_min = min(max(x_min, 0.0), 1.0)
    y_min = min(max(y_min, 0.0), 1.0)
    x_max = min(max(x_max, 0.0), 1.0)
    y_max = min(max(y_max, 0.0), 1.0)

    # 클리핑한 좌표를 x_center, y_center, width, height 형식으로 변환함
    clipped_x_center = (x_min + x_max) / 2
    clipped_y_center = (y_min + y_max) / 2
    clipped_width = x_max - x_min
    clipped_height = y_max - y_min

    return [clipped_x_center, clipped_y_center, clipped_width, clipped_height]

# 이미지와 바운딩박스 증강 수행
def augment_image_and_bboxes(image, bboxes, class_labels):
    # 파이프라인을 통해 증강 수행
    augmented = augmentation_pipeline(image=image, bboxes=bboxes, class_labels=class_labels)
    return augmented['image'], augmented['bboxes'], class_labels

# 증강 프로세스 실행
def augment_dataset():
    # 모든 이미지와 바운딩 박스 경로 수집
    image_bbox_pairs = []
    folder_mapping = {}
    for root, dirs, files in os.walk(base_dir):
        for dir_name in dirs:
            if augmentation_folder_suffix in dir_name:
                continue  # 이미 증강된 폴더는 건너뜀
            orig_dir = os.path.join(root, dir_name)
            aug_dir = os.path.join(root, dir_name + augmentation_folder_suffix)
            os.makedirs(aug_dir, exist_ok=True)
            folder_mapping[orig_dir] = aug_dir

            # 원본 폴더의 이미지와 라벨 쌍을 수집
            for file in os.listdir(orig_dir):
                if file.endswith(".jpg"):
                    img_path = os.path.join(orig_dir, file)
                    txt_path = os.path.join(orig_dir, file.replace(".jpg", ".txt"))
                    if os.path.exists(txt_path):  # 이미지와 txt 파일 쌍이 있을 때만 추가
                        image_bbox_pairs.append((img_path, txt_path, orig_dir))

    number_of_original_images = len(image_bbox_pairs)
    print(f"원본 이미지 수: {number_of_original_images}")

    # 필요한 증강 이미지 수 계산
    required_augmented_images = target_total_count - number_of_original_images
    if required_augmented_images <= 0:
        print("증강이 필요하지 않습니다. 이미 목표 수에 도달했습니다.")
        return

    # 이미지 경로 리스트를 무작위로 섞음
    random.shuffle(image_bbox_pairs)

    data_count = 0
    total_images = len(image_bbox_pairs)
    idx = 0

    # 증강 작업을 수행하며 목표 수에 도달할 때까지 반복함
    while data_count < required_augmented_images:
        if idx >= total_images:
            idx = 0  # 모든 이미지를 다 사용한 경우 리스트를 다시 섞어줌
            random.shuffle(image_bbox_pairs)
        img_path, txt_path, orig_dir = image_bbox_pairs[idx]
        aug_dir = folder_mapping[orig_dir]
        file_name = os.path.basename(img_path)
        aug_img_name = file_name.replace(".jpg", f"_aug_{data_count}.jpg")
        aug_txt_name = file_name.replace(".jpg", f"_aug_{data_count}.txt")
        aug_img_path = os.path.join(aug_dir, aug_img_name)
        aug_txt_path = os.path.join(aug_dir, aug_txt_name)

        # 이미지와 바운딩 박스 로드
        image = cv2.imread(img_path)
        if image is None:
            print(f"경고: 이미지를 불러올 수 없습니다. 파일을 건너뜁니다: {img_path}")
            idx += 1
            continue
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        bboxes_raw = load_bounding_boxes(txt_path)
        yolo_bboxes = [clip_bbox([float(b[1]), float(b[2]), float(b[3]), float(b[4])]) for b in bboxes_raw]
        class_labels = [b[0] for b in bboxes_raw]

        # 증강 적용
        aug_image, aug_bboxes, class_labels = augment_image_and_bboxes(image, yolo_bboxes, class_labels)

        # 증강된 이미지와 바운딩 박스를 저장
        aug_image_bgr = cv2.cvtColor(aug_image, cv2.COLOR_RGB2BGR)
        cv2.imwrite(aug_img_path, aug_image_bgr)
        aug_bboxes_norm = [[class_labels[i]] + list(bbox) for i, bbox in enumerate(aug_bboxes)]
        save_bounding_boxes(aug_txt_path, aug_bboxes_norm)

        data_count += 1
        idx += 1
        print(f"증강 완료: {aug_img_path}, 추가된 증강 이미지 수: {data_count}/{required_augmented_images}")

    print(f"증강 작업 완료: 총 이미지 수가 목표치({target_total_count}개)에 도달했습니다.")

# 증강 실행
augment_dataset()
