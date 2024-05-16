import os
import random
from PIL import Image
import imgaug.augmenters as iaa
import numpy as np

# 원본 이미지 경로
input_dir = r'C:\Users\SSTLabs\Desktop\1. POC촬영'
# 증강 이미지 저장 경로
output_dir = r'C:\Users\SSTLabs\Desktop\augmented_images'

# 증강 이미지 개수
num_augmented_images = 100

# 증강 기법 정의
seq = iaa.Sequential([
    iaa.Fliplr(0.5),  # 50% 확률로 좌우 반전
    iaa.Flipud(0.5),  # 50% 확률로 상하 반전
    iaa.Affine(rotate=(-30, 30)),  # -30도에서 30도 사이로 회전
    iaa.Multiply((0.8, 1.2)),  # 밝기 조정
])

# 이미지 파일 목록 가져오기
image_files = [f for f in os.listdir(input_dir) if f.endswith('.jpg')]

if not os.path.exists(output_dir):
    os.makedirs(output_dir)

# 이미지 증강 수행
for i in range(num_augmented_images):
    # 랜덤한 원본 이미지 선택
    image_file = random.choice(image_files)
    image_path = os.path.join(input_dir, image_file)
    image = Image.open(image_path)
    
    # imgaug를 사용하여 증강 수행
    image_aug = seq(image=np.array(image))
    
    # 증강된 이미지 저장
    output_path = os.path.join(output_dir, f'augmented_{i}.jpg')
    Image.fromarray(image_aug).save(output_path)

print(f'{num_augmented_images}개의 증강된 이미지를 {output_dir}에 저장했습니다.')
