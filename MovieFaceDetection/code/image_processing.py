import glob

import numpy as np
from skimage.feature import hog
import random
from Parameters import Parameters, is_test
import os
import cv2


def get_image_shapes(val_dir: str):
    img_shapes = {}
    if is_test:
        sub_folder = 'testare'
    else:
        sub_folder = 'validare'
    for img_path in glob.glob(os.path.join(val_dir, sub_folder, '*.jpg')):
        name = os.path.basename(img_path)
        img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
        img_shapes[name] = img.shape
    return img_shapes

def compute_iou(boxA, boxB):
    xA = max(boxA[0], boxB[0])
    yA = max(boxA[1], boxB[1])
    xB = min(boxA[2], boxB[2])
    yB = min(boxA[3], boxB[3])

    interW = max(0, xB - xA)
    interH = max(0, yB - yA)
    interArea = interW * interH

    areaA = (boxA[2] - boxA[0]) * (boxA[3] - boxA[1])
    areaB = (boxB[2] - boxB[0]) * (boxB[3] - boxB[1])

    union = areaA + areaB - interArea
    if union == 0:
        return 0

    return interArea / union


def pyramid(img, scale_factor_w=1.07, scale_factor_h=1.07, min_size=(36, 36)):
    """
    Generate images at different scales (pyramid)
    """
    h, w = img.shape
    min_h, min_w = min_size
    yield img, (1.0, 1.0), (w, h)
    # max_size = (800, 800)
    current_scale_w = 1.0
    current_scale_h = 1.0
    inv_scale_w = 1.0 / scale_factor_w
    inv_scale_h = 1.0 / scale_factor_h
    while True:
        current_scale_w *= inv_scale_w
        current_scale_h *= inv_scale_h
        new_w = int(w * current_scale_w)
        new_h = int(h * current_scale_h)

        if new_w < 2*min_w or new_h < 2*min_h:
            break

        resized_img = cv2.resize(img, (new_w, new_h), interpolation=cv2.INTER_AREA)
        yield resized_img, (current_scale_w, current_scale_h), (new_w, new_h)


class ImageProcessor:
    def __init__(self, params:Parameters):
        self.params = params

    def extract_hog(self, img):
        return hog(
            img,
            orientations=9,
            pixels_per_cell=(self.params.dim_hog_cell,
                             self.params.dim_hog_cell),
            cells_per_block=(2, 2),
            block_norm='L2-Hys',
            visualize=False,
            feature_vector=True
        )

    def get_positive_descriptor(self, image, bbox, with_color=False):
        x1, y1, x2, y2 = bbox
        if x1 < 0 or x1 > image.shape[1] or y1 < 0 or y1 > image.shape[0]:
            return None
        face = image[y1:y2, x1:x2]

        if face.size == 0:
            return None

        face = cv2.resize(face, self.params.dim_face)
        gray_face = cv2.cvtColor(face, cv2.COLOR_BGR2GRAY)
        hog_desc = self.extract_hog(gray_face)
        if hog_desc is None:
            return None
        if with_color:
            hsv_face = cv2.cvtColor(face, cv2.COLOR_BGR2HSV)
            h_hist = cv2.calcHist([hsv_face], [0], None, [self.params.hist_size], [0, 180])
            s_hist = cv2.calcHist([hsv_face], [1], None, [self.params.hist_size], [0, 256])
            v_hist = cv2.calcHist([hsv_face], [2], None, [self.params.hist_size], [0, 256])
            color_feat = np.concatenate([h_hist, s_hist, v_hist]).flatten()
            color_feat = color_feat / (np.sum(color_feat) + 1e-7)
            hybrid_feat = np.concatenate([hog_desc, color_feat])
            return hybrid_feat

        return hog_desc

    def get_negative_descriptors(self, image, face_bboxes, num_samples):
        h, w = image.shape
        gray_img = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        neg_desc = []
        attempts = 0
        max_attempts = num_samples * 10

        while len(neg_desc) < num_samples and attempts < max_attempts:
            attempts += 1

            x = random.randint(0, w - self.params.dim_face[0])
            y = random.randint(0, h - self.params.dim_face[1])

            neg_box = [x, y,
                       x + self.params.dim_face[0],
                       y + self.params.dim_face[1]]

            overlap = False
            for fb in face_bboxes:
                if compute_iou(neg_box, fb) > 0:
                    overlap = True
                    break

            if overlap:
                continue

            patch = gray_img[y:y + self.params.dim_face[1],
                    x:x + self.params.dim_face[0]]

            # neg_file_dir = os.path.join(self.params.dir_save_files, 'negative_img_samples')
            # cur_id = len(os.listdir(neg_file_dir))
            #cv2.imwrite(os.path.join(neg_file_dir, f'neg_{cur_id}.png'), patch)

            hog_desc = self.extract_hog(patch)

            neg_desc.append(hog_desc)

        return neg_desc

    def generate_train_data(self, with_negative=False, with_color = False):
        pos_data = []
        neg_data = []
        pos_labels = []
        prev_img_name = ''
        img = None
        for pos_dir_name in self.params.pos_dir_names:
            file_name = f'{pos_dir_name}_annotations.txt'
            annot_path = os.path.join(self.params.train_dir, file_name)
            face_bboxes = []
            with open(annot_path, 'r') as f:
                for line in f:
                    content = line.strip().split(' ')
                    img_name = content[0]
                    if img_name != prev_img_name:
                        if with_negative:
                            if len(prev_img_name) > 0:
                                neg_descs = self.get_negative_descriptors(img, face_bboxes, 5)
                                neg_data.extend(neg_descs)
                        prev_img_name = img_name
                        face_bboxes = []
                        img_path = os.path.join(self.params.train_dir, pos_dir_name, img_name)
                        img = cv2.imread(str(img_path), cv2.IMREAD_COLOR_BGR)
                    label = content[-1]
                    if label == 'unknown':
                        continue
                    label_idx = self.params.pos_dir_names.index(label)
                    bbox = list(map(int, content[1:5]))
                    hog_desc = self.get_positive_descriptor(img, bbox, with_color=with_color)
                    if hog_desc is not None:
                        pos_data.append(hog_desc)
                        pos_labels.append(label_idx)
                    face_bboxes.append(bbox)
                    padding = 7
                    bbox[0] -= padding
                    bbox[1] -= padding
                    bbox[2] += padding
                    bbox[3] += padding
                    hog_desc = self.get_positive_descriptor(img, bbox, with_color=with_color)
                    if hog_desc is not None:
                        pos_data.append(hog_desc)
                        pos_labels.append(label_idx)
        return np.array(pos_data), np.array(neg_data), np.array(pos_labels)

    def save_descriptors(self, pos_data, neg_data, with_color=False):
        color = ''
        if with_color:
            color = 'color'
        pos_save_path = os.path.join(self.params.dir_save_files, 'descriptors', f'pos_descriptors_{self.params.dim_face[0]}_{self.params.dim_face[1]}{color}.npy')
        neg_save_path = os.path.join(self.params.dir_save_files, 'descriptors', f'neg_descriptors_{self.params.dim_face[0]}_{self.params.dim_face[1]}{color}.npy')
        if len(pos_data) > 0:
            np.save(pos_save_path, pos_data)
            print('Saved positive descriptors')
        if len(neg_data) > 0:
            np.save(neg_save_path, neg_data)
            print('Saved negative descriptors')

    def load_descriptors(self, with_color=False):
        color = ''
        if with_color:
            color = 'color'
        pos_data_path = os.path.join(self.params.dir_save_files, 'descriptors', f'pos_descriptors_{self.params.dim_face[0]}_{self.params.dim_face[1]}{color}.npy')
        neg_data_path = os.path.join(self.params.dir_save_files, 'descriptors', f'neg_descriptors_{self.params.dim_face[0]}_{self.params.dim_face[1]}{color}.npy')
        if not os.path.exists(pos_data_path):
            return None, None
        pos_data = np.load(pos_data_path)
        if not os.path.exists(neg_data_path):
            neg_data = None
        else:
            neg_data = np.load(neg_data_path)
        return pos_data, neg_data

    def get_pos_labels(self):
        pos_labels = []
        for pos_dir_name in self.params.pos_dir_names:
            file_name = f'{pos_dir_name}_annotations.txt'
            annot_path = os.path.join(self.params.train_dir, file_name)
            with open(annot_path, 'r') as f:
                for line in f:
                    content = line.strip().split(' ')
                    label = content[-1]
                    if label == 'unknown':
                        label_idx = -1
                    else:
                        label_idx = self.params.pos_dir_names.index(label)
                    pos_labels.append(label_idx)
                    pos_labels.append(label_idx)
        return np.array(pos_labels)
