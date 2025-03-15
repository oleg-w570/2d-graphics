import os
import random
import glob
import cv2
import numpy as np
import matplotlib.pyplot as plt
from typing import List, Tuple, Dict
import json
import torch
from torch.utils.data import Dataset

class VisDroneDataset(Dataset):
    """Dataset для генерации изображений с объектами из VisDrone"""

    def __init__(
        self,
        dataset_path: str,
        img_size: Tuple[int, int] = (800, 800),
        dataset_size: int = 1000,
        max_objects: int = 10
    ):
        self.dataset_path = dataset_path
        self.background_path = os.path.join(dataset_path, 'images')
        self.annotations_path = os.path.join(dataset_path, 'annotations')
        self.background_images = glob.glob(os.path.join(self.background_path, '*.jpg'))
        self.image_size = img_size
        self.dataset_size = dataset_size
        self.max_objects = max_objects

    def __len__(self):
        return self.dataset_size

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, List[Dict], int]:
        """
        Генерация одного изображения с объектами
        
        Returns:
            Tuple[torch.Tensor, List[Dict], int]: (изображение, список bbox'ов, количество объектов)
        """
        # Случайное количество объектов
        num_objects = random.randint(0, self.max_objects)
        
        # Создание фона
        background = self._create_background(self.image_size)
        h, w = background.shape[:2]
        
        # Список для хранения bbox
        bboxes = []
        
        # Выбор случайного изображения с объектами
        obj_img_path = random.choice(self.background_images)
        obj_img = cv2.imread(obj_img_path)
        obj_annotations = self._load_annotation(os.path.basename(obj_img_path))
        
        # Фильтрация только автомобилей (категория 4)
        car_annotations = [ann for ann in obj_annotations if ann['category'] == 4]
        
        for _ in range(num_objects):
            if not car_annotations:
                continue
                
            # Выбор случайного автомобиля
            car_ann = random.choice(car_annotations)
            x, y, w_obj, h_obj = map(int, car_ann['bbox'])
            
            # Извлечение патча с автомобилем
            car_patch = obj_img[y:y+h_obj, x:x+w_obj]
            
            # Случайное масштабирование
            scale = random.uniform(0.5, 1.5)
            car_patch = cv2.resize(car_patch, None, fx=scale, fy=scale)
            
            # Случайное положение
            max_x = w - car_patch.shape[1]
            max_y = h - car_patch.shape[0]
            if max_x <= 0 or max_y <= 0:
                continue
                
            rand_x = random.randint(0, max_x)
            rand_y = random.randint(0, max_y)
            
            # Проверка перекрытия
            overlap = False
            for bbox in bboxes:
                if self._calculate_overlap(
                    (rand_x, rand_y, car_patch.shape[1], car_patch.shape[0]),
                    bbox
                ) > 0.5:
                    overlap = True
                    break
                    
            if overlap:
                continue
                
            # Цветокоррекция
            ref_patch = background[rand_y:rand_y+car_patch.shape[0], 
                                 rand_x:rand_x+car_patch.shape[1]]
            car_patch = self._color_correct_patch(car_patch, ref_patch)
            
            # Наложение с использованием seamless cloning
            mask = np.ones(car_patch.shape[:2], dtype=np.uint8) * 255
            center = (rand_x + car_patch.shape[1]//2, 
                     rand_y + car_patch.shape[0]//2)
            background = cv2.seamlessClone(car_patch, background, mask, center, 
                                         cv2.NORMAL_CLONE)
            
            # Сохранение bbox
            bboxes.append({
                'bbox': [rand_x, rand_y, car_patch.shape[1], car_patch.shape[0]],
                'category': 4
            })
            
        # Конвертация в тензор PyTorch
        img_tensor = torch.from_numpy(background).float().permute(2, 0, 1)
        
        return img_tensor, bboxes, len(bboxes)

    def _load_annotation(self, img_name: str) -> List[Dict]:
        """Загрузка аннотаций для изображения"""
        ann_path = os.path.join(self.annotations_path, 
                               os.path.splitext(img_name)[0] + '.txt')
        if not os.path.exists(ann_path):
            return []
            
        annotations = []
        with open(ann_path, 'r') as f:
            for line in f:
                # Формат аннотации: <bbox_left>,<bbox_top>,<bbox_width>,<bbox_height>,<score>,<object_category>,<truncation>,<occlusion>
                values = line.strip().split(',')
                if len(values) >= 8:  # Проверяем, что есть все необходимые значения
                    x, y, w, h, _, category, _, _ = map(float, values[:8])
                    annotations.append({
                        'bbox': [x, y, w, h],
                        'category': int(category)
                    })
        return annotations
    
    def _create_background(self, size: Tuple[int, int]) -> np.ndarray:
        """Создание фонового изображения из 4 случайных патчей"""
        h, w = size[0] // 2, size[1] // 2
        background = np.zeros((size[0], size[1], 3), dtype=np.uint8)
        
        # Выбор случайных изображений для каждой ячейки сетки
        selected_bg = random.sample(self.background_images, 4)
        bg_imgs = [cv2.imread(p) for p in selected_bg]
        
        # Размещение изображений в сетке 2x2
        background[0:h, 0:w] = cv2.resize(bg_imgs[0], (w, h))
        background[0:h, w:2*w] = cv2.resize(bg_imgs[1], (w, h))
        background[h:2*h, 0:w] = cv2.resize(bg_imgs[2], (w, h))
        background[h:2*h, w:2*w] = cv2.resize(bg_imgs[3], (w, h))
        
        return background
    
    def _color_correct_patch(self, src: np.ndarray, ref: np.ndarray) -> np.ndarray:
        """Статистическая цветокоррекция патча"""
        src = src.astype(float)
        ref = ref.astype(float)
        
        for c in range(3):
            src_mean, src_std = src[..., c].mean(), src[..., c].std() + 1e-5
            ref_mean, ref_std = ref[..., c].mean(), ref[..., c].std() + 1e-5
            src[..., c] = ((src[..., c] - src_mean) * (ref_std / src_std) + ref_mean)
            
        return np.clip(src, 0, 255).astype(np.uint8)
    
    def _calculate_overlap(self, bbox1: Tuple[int, int, int, int], 
                          bbox2: Dict) -> float:
        """Вычисление доли перекрытия двух bbox"""
        x1, y1, w1, h1 = bbox1
        x2, y2, w2, h2 = bbox2['bbox']
        
        # Вычисление площади пересечения
        x_left = max(x1, x2)
        y_top = max(y1, y2)
        x_right = min(x1 + w1, x2 + w2)
        y_bottom = min(y1 + h1, y2 + h2)
        
        if x_right < x_left or y_bottom < y_top:
            return 0.0
            
        intersection_area = (x_right - x_left) * (y_bottom - y_top)
        bbox1_area = w1 * h1
        bbox2_area = w2 * h2
        
        return intersection_area / float(bbox1_area + bbox2_area - intersection_area)

# Пример использования
if __name__ == "__main__":
    # Создание датасета
    dataset = VisDroneDataset('visdrone-dataset', dataset_size=32)
    
    # Визуализация нескольких примеров
    figure = plt.figure(figsize=(15, 10))
    cols, rows = 3, 2
    
    for i in range(1, cols * rows + 1):
        sample_idx = random.randint(0, len(dataset) - 1)
        img, bboxes, num_objects = dataset[sample_idx]
        
        # Конвертация тензора обратно в изображение для отображения
        img_np = img.permute(1, 2, 0).numpy().astype(np.uint8)
        
        ax = figure.add_subplot(rows, cols, i)
        ax.imshow(cv2.cvtColor(img_np, cv2.COLOR_BGR2RGB))
        ax.set_title(f'Объектов: {num_objects}')
        
        # Отрисовка bbox
        for bbox in bboxes:
            x, y, w, h = bbox['bbox']
            rect = plt.Rectangle((x, y), w, h, fill=False, color='red', linewidth=2)
            ax.add_patch(rect)
            
        ax.axis('off')
    
    plt.tight_layout()
    plt.show()
    
    # Пример гистограммы для одного изображения
    plt.figure(figsize=(8, 4))
    plt.hist(img_np.ravel(), bins=256, color='orange', alpha=0.7)
    plt.title('Гистограмма изображения')
    plt.show()
