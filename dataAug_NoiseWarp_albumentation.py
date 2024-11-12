import os
import cv2
import numpy as np
import random
from albumentations import (
    Compose, OneOf, RandomBrightnessContrast, HueSaturationValue,
    GaussNoise, Rotate, Flip, ShiftScaleRotate, ElasticTransform, GridDistortion
)
from PIL import Image

# 경로 설정
base_dir = r"E:\GitHub\HJ\241101stampAug"  # 원본 이미지 폴더 경로 설정
target_total_count = 3000  # 목표로 하는 총 이미지 개수 (원본 + 증강)

augmentation_folder_suffix = "_aug"  # 증강 이미지 저장 폴더 접미사

# 증강 파이프라인 설정 (3가지 사례를 적용하여 다양한 이미지 증강)
augmentation_pipeline = Compose([
    # 사례 1: 조명 조건 변화 시뮬레이션 - 밝기와 색상을 변경하여 다양한 조명 환경을 학습하도록 함
    OneOf([
        RandomBrightnessContrast(brightness_limit=0.2, contrast_limit=0.2, p=0.5),
        # 밝기 및 대비 조정 - 다양한 조도 상황에서의 성능 향상을 위해 조정함
        HueSaturationValue(hue_shift_limit=10, sat_shift_limit=20, val_shift_limit=10, p=0.5),
        # 색조, 채도, 밝기를 변경 - 색상 변화가 있는 상황에서도 강건하게 동작하도록 학습함
    ], p=0.7),
    
    # 사례 2: 기하학적 변환 - 회전, 반전, 이동 및 스케일링을 통해 다양한 위치와 방향에서의 인식을 가능하게 함
    Rotate(limit=15, p=0.7),
    # 15도 내외로 이미지를 회전시켜 다양한 각도에서도 모델이 인식할 수 있게 함
    Flip(p=0.5),
    # 좌우 반전 - 좌우 방향이 바뀌어도 모델이 제대로 인식하도록 함
    ShiftScaleRotate(shift_limit=0.1, scale_limit=0.1, rotate_limit=0, p=0.5),
    # 이동 및 스케일 변환 - 객체가 중심에 있지 않거나 크기가 다를 때도 인식 가능하게 함
    
    # 사례 3: 노이즈 및 왜곡 적용 - 다양한 노이즈와 변형을 추가해 이미지 왜곡 상황에 강건하게 학습시킴
    OneOf([
        GaussNoise(var_limit=(10.0, 50.0), p=0.3),
        # 가우스 노이즈 추가 - 촬영 환경에서 발생할 수 있는 잡음 상황을 학습시킴
        ElasticTransform(alpha=1, sigma=50, alpha_affine=50, p=0.5),
        # 탄성 변형 - 이미지를 휘게 하여 비정형 왜곡에도 대응할 수 있도록 함
        GridDistortion(num_steps=5, distort_limit=0.3, p=0.5),
        # 격자 왜곡 - 격자 형태로 왜곡을 주어 다양한 패턴의 왜곡을 학습하도록 함
    ], p=0.5),
    
], bbox_params={'format': 'yolo', 'label_fields': ['class_labels'], 'min_visibility': 0.1})

# 바운딩박스 로드 및 저장 함수
def load_bounding_boxes(txt_path):
    # 텍스트 파일에서 바운딩 박스 정보를 읽어옴
    with open(txt_path, "r") as f:
        lines = f.readlines()
    return [line.strip().split() for line in lines]

def save_bounding_boxes(txt_path, bboxes):
    # 바운딩 박스를 YOLO 형식으로 저장
    with open(txt_path, "w") as f:
        for bbox in bboxes:
            f.write(" ".join(map(str, bbox)) + "\n")

# 바운딩 박스 좌표를 클리핑하여 [0, 1] 범위로 제한하는 함수
def clip_bbox(bbox):
    x_center, y_center, width, height = bbox
    x_min = x_center - width / 2
    y_min = y_center - height / 2
    x_max = x_center + width / 2
    y_max = y_center + height / 2

    # x_min, y_min, x_max, y_max를 [0.0, 1.0] 범위 내로 클리핑
    x_min = min(max(x_min, 0.0), 1.0)
    y_min = min(max(y_min, 0.0), 1.0)
    x_max = min(max(x_max, 0.0), 1.0)
    y_max = min(max(y_max, 0.0), 1.0)

    # 클리핑된 좌표를 x_center, y_center, width, height 형식으로 반환
    clipped_x_center = (x_min + x_max) / 2
    clipped_y_center = (y_min + y_max) / 2
    clipped_width = x_max - x_min
    clipped_height = y_max - y_min

    return [clipped_x_center, clipped_y_center, clipped_width, clipped_height]

# 이미지와 바운딩 박스를 증강
def augment_image_and_bboxes(image, bboxes, class_labels):
    # 증강 파이프라인을 적용하여 이미지와 바운딩 박스를 증강함
    augmented = augmentation_pipeline(image=image, bboxes=bboxes, class_labels=class_labels)
    return augmented['image'], augmented['bboxes'], class_labels

# 증강 프로세스를 수행
def augment_dataset():
    # 모든 이미지와 바운딩 박스 경로를 수집
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

            for file in os.listdir(orig_dir):
                if file.endswith(".jpg"):
                    img_path = os.path.join(orig_dir, file)
                    txt_path = os.path.join(orig_dir, file.replace(".jpg", ".txt"))
                    if os.path.exists(txt_path):  # 이미지와 txt 쌍이 있을 때만 추가
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

    # 증강 이미지가 목표 수에 도달할 때까지 반복 수행
    while data_count < required_augmented_images:
        if idx >= total_images:
            idx = 0  # 모든 이미지를 다 사용한 경우 다시 섞어줌
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

        # 증강된 이미지 및 바운딩박스 저장
        aug_image_bgr = cv2.cvtColor(aug_image, cv2.COLOR_RGB2BGR)
        cv2.imwrite(aug_img_path, aug_image_bgr)
        aug_bboxes_norm = [[class_labels[i]] + list(bbox) for i, bbox in enumerate(aug_bboxes)]
        save_bounding_boxes(aug_txt_path, aug_bboxes_norm)

        data_count += 1
        idx += 1
        print(f"증강 완료: {aug_img_path}, 추가된 증강 이미지 수: {data_count}/{required_augmented_images}")

   
