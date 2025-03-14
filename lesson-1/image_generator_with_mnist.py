# %%
import torch
from torch.utils.data import Dataset
from torchvision import datasets
import torchvision.transforms as v2
import matplotlib.pyplot as plt


training_data = datasets.FashionMNIST(
    root="data",
    train=True,
    download=True,
    transform=v2.ToTensor())
)

# %%
labels_map = {
    0: "T-Shirt",
    1: "Trouser",
    2: "Pullover",
    3: "Dress",
    4: "Coat",
    5: "Sandal",
    6: "Shirt",
    7: "Sneaker",
    8: "Bag",
    9: "Ankle Boot",
}
figure = plt.figure(figsize=(8, 8))
cols, rows = 3, 3
for i in range(1, cols * rows + 1):
    sample_idx = torch.randint(len(training_data), size=(1,)).item()
    img, label = training_data[sample_idx]
    figure.add_subplot(rows, cols, i)
    plt.title(labels_map[label])
    plt.axis("off")
    plt.imshow(img.squeeze(), cmap="gray")
plt.show()


# %%
from __future__ import annotations

import numpy as np
import cv2
import copy
from typing import Tuple
import random
import torch
from torch.utils.data import Dataset

class ChessBoardWildFashionDataset(Dataset):
    """Chess board dataset with MNIST squares as defects"""

    def __init__(
        self,
        img_size: Tuple[int, int],
        patch_size: Tuple[int, int],
        dataset_size: int,
        noise_dataset: Dataset,  # "mnist"
    ):
        self.image_size = img_size
        self.ps = patch_size
        self.img_base = np.zeros((self.ps[0] * 8, self.ps[1] * 8, 3), dtype=np.uint8)
        self.dataset_size = dataset_size
        self.noise_images = noise_dataset

        for i in range(8):
            for j in range(8):
                if (i + j) % 2 == 0:
                    self.img_base[i * self.ps[0] : (i + 1) * self.ps[0], j * self.ps[1] : (j + 1) * self.ps[1], :] = (
                        np.ones((self.ps[0], self.ps[1], 3), dtype=np.uint8) * 255
                    )

    def __len__(self):
        return self.dataset_size

    def __getitem__(self, item: int) -> Tuple[torch.Tensor, torch.Tensor]:  # image, 17-class target 1:
        img = copy.deepcopy(self.img_base)

        count = random.randint(0, 11)  # count of defects

        # label list
        # [0] - 0/1 - not rejected/rejected image
        # [1:17] - 0/1 - note defected/defected image patch
        labels = [0 for n in range(17)]
        plus_i = 1

        # if defect count > 5, the image is rejected
        if count > 5:
            labels[0] = 1
        for i in range(count):
            # randomly choose defect image from noise dataset
            img_num,_ = self.noise_images[random.randint(0, len(self.noise_images) - 1)]
            img_num = v2.Resize((32,32))(img_num)*2
            img_num = np.repeat(np.reshape(img_num.numpy(), (32,32,1)),3,axis = 2)

            # randomly choose defect location
            i = random.randint(0, 7)
            j = random.randint(0, 7)

            idx = i // 2 * 4 + j // 2
            labels[idx + plus_i] = 1

            img[i * self.ps[0] : (i + 1) * self.ps[0], j * self.ps[1] : (j + 1) * self.ps[1], :] = img_num

        img = cv2.resize(img, dsize=(self.image_size[0], self.image_size[1]))
        img = img.astype(float)
        #img /= 255  # normalize data
        
        return (
            torch.FloatTensor(img),
            sum(labels[1:]),
        )

# %%
cbwf_dataset =  ChessBoardWildFashionDataset((256,256),(32,32),32, training_data)

# %%
labels_map = {
    0: '0',
    1: '1',
    2: '2',
    3: '3',
    4: '4',
    5: '5',
    6: '6',
    7: '7',
    8: '8',
    9: '9',
    10: '10',
}
figure = plt.figure(figsize=(8, 8))
cols, rows = 3, 3
for i in range(1, cols * rows + 1):
    sample_idx = torch.randint(len(cbwf_dataset), size=(1,)).item()
    img, label = cbwf_dataset[sample_idx]
    figure.add_subplot(rows, cols, i)
    plt.title(labels_map[label])
    plt.axis("off")
    plt.imshow(img.squeeze(), cmap="gray")
plt.show()

# %%
# 1.1 Генерация данных
import os
import random
import glob
import cv2
import numpy as np
# Загрузка случайных четырёх изображений из visdrone-dataset
bg_paths = glob.glob('/workspaces/2d-graphics/visdrone-dataset/images/*.jpg')
selected_bg = random.sample(bg_paths, 4)
bg_imgs = [cv2.imread(p) for p in selected_bg]
# Сетка 2x2
h, w, _ = bg_imgs[0].shape
combined_bg = np.zeros((2*h, 2*w, 3), dtype=np.uint8)
combined_bg[0:h, 0:w] = bg_imgs[0]
combined_bg[0:h, w:2*w] = bg_imgs[1]
combined_bg[h:2*h, 0:w] = bg_imgs[2]
combined_bg[h:2*h, w:2*w] = bg_imgs[3]
# 1.2 Препроцессинг патчей (пример простой цветокоррекции)
def color_correct_patches(src, ref):
    # Быстрая коррекция: приводим среднее и std по каналам
    for c in range(3):
        src_mean, src_std = src[..., c].mean(), src[..., c].std()+1e-5
        ref_mean, ref_std = ref[..., c].mean(), ref[..., c].std()+1e-5
        src[..., c] = ((src[..., c] - src_mean) * (ref_std / src_std) + ref_mean)
    return np.clip(src, 0, 255).astype(np.uint8)
# Пример случайного наложения одного объекта
obj_paths = glob.glob('/workspaces/2d-graphics/visdrone-dataset/objects/*.jpg')
obj_img = cv2.imread(random.choice(obj_paths))
obj_h, obj_w, _ = obj_img.shape
# Цветокоррекция объекта по фоновому патчу
patch_for_ref = combined_bg[h//2:h, w//2:w, :]
obj_img = color_correct_patches(obj_img, patch_for_ref)
# Случайные координаты наложения (учитываем ограничение перекрытия)
max_x = combined_bg.shape[1] - obj_w
max_y = combined_bg.shape[0] - obj_h
rand_x = random.randint(0, max_x)
rand_y = random.randint(0, max_y)
combined_bg[rand_y:rand_y+obj_h, rand_x:rand_x+obj_w] = obj_img
# Количество объектов (пример)
object_count = 1
# 1.3 Визуализация
import matplotlib.pyplot as plt
fig, axs = plt.subplots(1, 2, figsize=(10,5))
axs[0].imshow(cv2.cvtColor(combined_bg, cv2.COLOR_BGR2RGB))
axs[0].set_title(f'Objects: {object_count}')
axs[0].axis('off')
# Гистограмма
axs[1].hist(combined_bg.ravel(), bins=256, color='orange', alpha=0.7)
axs[1].set_title('Гистограмма изображения')
plt.show()


