import os
import cv2
import numpy as np
import random
from PIL import Image, ImageStat, ImageOps, ImageFilter

# 특정 강도의 블러 효과를 이미지에 적용함
def apply_blur(image, blur_strength):
    return image.filter(ImageFilter.GaussianBlur(blur_strength))

# 이미지의 평균 색상(RGB)을 계산하는 함수
def get_average_color(image):
    stat = ImageStat.Stat(image)
    r, g, b = stat.mean[:3]
    return (r, g, b)

# 알파 채널이 적용된 이미지에서 특정 알파 범위 내 평균 색상 추출
def get_alpha_average_color(image):
    pixels = np.array(image)
    mask = (pixels[..., 3] >= 128) & (pixels[..., 3] <= 240)  # 알파값이 50% 이상 90% 이하인 부분만 고려
    if np.any(mask):
        avg_color = np.mean(pixels[mask], axis=0)[:3]
        return tuple(avg_color)
    else:
        return get_average_color(image)

# 유사한 색상을 가진 영역을 찾기 위한 함수
def find_similar_color_region(image, piece_image, margin, edge_mask, bright_mask):
    image_width, image_height = image.size
    piece_width, piece_height = piece_image.size
    piece_avg_color = get_alpha_average_color(piece_image)
    min_diff = float('inf')  # 최소 색상 차이를 추적함
    best_coords = (0, 0)  # 최적의 좌표 초기화

    # 이미지를 256등분하여 평균 색상 비교
    num_divisions = 16
    step_x = (image_width - 2 * margin - piece_width) // num_divisions
    step_y = (image_height - 2 * margin - piece_height) // num_divisions

    for i in range(num_divisions):
        for j in range(num_divisions):
            x = margin + i * step_x
            y = margin + j * step_y

            # 이미지 범위 내의 영역만 탐색
            if x + piece_width > image_width or y + piece_height > image_height:
                continue
            region = image.crop((x, y, x + piece_width, y + piece_height))
            region_avg_color = get_average_color(region)

            # 밝기, 엣지 마스크 조건을 만족하며 조각 평균보다 어두운 영역 찾기
            if np.mean(region_avg_color) < np.mean(piece_avg_color) + 5 and \
               not np.any(edge_mask[y:y + piece_height, x:x + piece_width]) and \
               not np.any(bright_mask[y:y + piece_height, x:x + piece_width]):
                diff = np.linalg.norm(np.array(piece_avg_color) - np.array(region_avg_color))
                if diff < min_diff:
                    min_diff = diff
                    best_coords = (x, y)

    return best_coords

# 이미지의 평균 밝기 값을 반환
def get_light_pattern(image):
    gray_image = ImageOps.grayscale(image)
    stat = ImageStat.Stat(gray_image)
    return stat.mean[0]

# YOLO 형식의 라벨 생성 (중심 좌표와 크기 비율로 계산)
def create_yolo_label(class_id, image_width, image_height, piece_width, piece_height, x, y):
    center_x = (x + piece_width / 2) / image_width
    center_y = (y + piece_height / 2) / image_height
    width = piece_width / image_width
    height = piece_height / image_height
    return f"{class_id} {center_x} {center_y} {width} {height}"

# 알파 채널 값을 조정하는 함수
def adjust_alpha(image, alpha_increment):
    r, g, b, a = image.split()
    a = a.point(lambda i: max(i + alpha_increment, 0))
    return Image.merge("RGBA", (r, g, b, a))

# 이미지에서 엣지 영역을 감지하고 확장함
def detect_edges(image):
    gray_image = cv2.cvtColor(np.array(image), cv2.COLOR_RGBA2GRAY)
    edges = cv2.Canny(gray_image, 100, 200)
    kernel = np.ones((5, 5), np.uint8)
    edges = cv2.dilate(edges, kernel, iterations=1)  # 엣지 영역 확장
    return edges

# 이미지에서 밝은 영역을 감지하는 함수 (기본 임계값 200)
def detect_bright_areas(image, threshold=200):
    gray_image = cv2.cvtColor(np.array(image), cv2.COLOR_RGBA2GRAY)
    _, bright_areas = cv2.threshold(gray_image, threshold, 255, cv2.THRESH_BINARY)
    return bright_areas

# 원본 이미지에 조각 이미지를 합성하고 라벨 생성
def process_image(original_image_path, piece_images, pieces_folder, output_folder, label_folder, original_folder, class_id):
    try:
        original_image = Image.open(original_image_path).convert("RGBA")
        print(f"Opened original image: {original_image_path}")
    except Exception as e:
        print(f"Failed to open image {original_image_path}: {e}")
        return
    image_width, image_height = original_image.size

    # 랜덤으로 조각 이미지 선택
    piece_image_name = random.choice(piece_images)
    piece_image_path = os.path.join(pieces_folder, piece_image_name)
    try:
        piece_image = Image.open(piece_image_path).convert("RGBA")
        print(f"Opened piece image: {piece_image_path}")
    except Exception as e:
        print(f"Failed to open piece image {piece_image_path}: {e}")
        return

    # 알파값 감소로 투명도 조정
    piece_image = adjust_alpha(piece_image, -2)

    # 엣지 및 밝은 영역 마스크 생성
    edge_mask = detect_edges(original_image)
    bright_mask = detect_bright_areas(original_image)

    # 적합한 위치에 조각 이미지 합성
    margin = 10
    x, y = find_similar_color_region(original_image, piece_image, margin, edge_mask, bright_mask)

    # 빛의 패턴에 따라 이미지 미러링
    piece_light_pattern = get_light_pattern(piece_image)
    region = original_image.crop((x, y, x + piece_image.width, y + piece_image.height))
    region_light_pattern = get_light_pattern(region)

    # 필요 시 블러링 적용
    piece_image = apply_blur(piece_image, 0.2)
    if piece_light_pattern > region_light_pattern:
        piece_image = ImageOps.mirror(piece_image)

    # 최종 합성
    original_image.paste(piece_image, (x, y), piece_image)

    # YOLO 형식 라벨 생성
    label = create_yolo_label(class_id, image_width, image_height, piece_image.width, piece_image.height, x, y)
    labels = [label]

    # 상대 경로와 출력 경로 생성
    relative_path = os.path.relpath(original_image_path, original_folder)
    output_image_path = os.path.join(output_folder, relative_path)
    base_name = os.path.splitext(relative_path)[0]
    output_label_path = os.path.join(label_folder, base_name + '.txt')

    # 출력 디렉토리 생성 및 저장
    os.makedirs(os.path.dirname(output_image_path), exist_ok=True)
    os.makedirs(os.path.dirname(output_label_path), exist_ok=True)

    try:
        if original_image_path.lower().endswith(('.jpg', '.jpeg')):
            original_image_to_save = original_image.convert("RGB")
        else:
            original_image_to_save = original_image
        original_image_to_save.save(output_image_path)
    except Exception as e:
        print(f"Failed to save image {output_image_path}: {e}")

    # 기존 라벨 파일 읽기
    original_label_path = os.path.join(original_folder, base_name + '.txt')
    existing_labels = []
    if os.path.exists(original_label_path):
        try:
            with open(original_label_path, 'r') as f:
                existing_labels = f.read().splitlines()
        except Exception as e:
            print(f"Failed to read label file {original_label_path}: {e}")

    # 새로운 라벨 추가 후 저장
    existing_labels.append(label)
    try:
        with open(output_label_path, 'w') as label_file:
            label_file.write('\n'.join(existing_labels))
    except Exception as e:
        print(f"Failed to write label file {output_label_path}: {e}")

# 메인 함수 실행
def main():
    original_folder = r"E:\GitHub\HJ\p4"  # 원본 이미지 폴더
    pieces_folder = r"E:\GitHub\HJ\P1_BUBBLE_NUKKI"  # 조각 이미지 폴더
    output_folder = r"E:\GitHub\HJ\p4_output"  # 출력 폴더
    label_folder = output_folder  # 라벨 파일 폴더 
    class_id = 2  # 클래스 ID

    os.makedirs(output_folder, exist_ok=True)
    os.makedirs(label_folder, exist_ok=True)

    # 조각 이미지 리스트 생성
    piece_images = [f for f in os.listdir(pieces_folder) if f.lower().endswith('png')]

    # 원본 폴더의 이미지 처리
    for root, _, files in os.walk(original_folder):
        for file in files:
            if file.lower().endswith(('jpg', 'jpeg')):
                original_image_path = os.path.join(root, file)
                process_image(original_image_path, piece_images, pieces_folder, output_folder, label_folder, original_folder, class_id)

if __name__ == "__main__":
    main()
