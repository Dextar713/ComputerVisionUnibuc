import torch
import os
import cv2
import numpy as np
import yaml
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms
from MovieFaceDetection.code.Parameters import Parameters, is_test
from ultralytics import YOLO
from MovieFaceDetection.code.FacialDetector import FacialDetector


def prepare_data(params: Parameters):
    # Process each character folder
    for pos_dir_name in params.pos_dir_names:
        file_name = f'{pos_dir_name}_annotations.txt'
        annot_path = os.path.join(params.train_dir, file_name)

        # Use a dictionary to group annotations by image name
        img_annotations = {}
        if not os.path.exists(annot_path):
            continue

        with open(annot_path, 'r') as f:
            for line in f:
                content = line.strip().split(' ')
                img_name = content[0]
                label_name = content[-1]
                if label_name == 'unknown':
                    continue

                label_idx = params.pos_dir_names.index(label_name)
                bbox = list(map(int, content[1:5]))

                if img_name not in img_annotations:
                    img_annotations[img_name] = []
                img_annotations[img_name].append((label_idx, bbox))

        # Write labels directly into the character's image folder
        # This ensures YOLO finds them immediately
        char_img_folder = os.path.join(params.train_dir, pos_dir_name)

        for img_name, annotations in img_annotations.items():
            img_path = os.path.join(char_img_folder, img_name)
            img = cv2.imread(img_path)
            if img is None:
                continue
            h_orig, w_orig = img.shape[:2]

            # Save .txt in the SAME folder as the .jpg
            txt_name = os.path.splitext(img_name)[0] + '.txt'
            write_path = os.path.join(char_img_folder, txt_name)

            with open(write_path, 'w') as f_out:
                for label_idx, bbox in annotations:
                    x_center = (bbox[0] + bbox[2]) / (2 * w_orig)
                    y_center = (bbox[1] + bbox[3]) / (2 * h_orig)
                    w = (bbox[2] - bbox[0]) / w_orig
                    h = (bbox[3] - bbox[1]) / h_orig
                    f_out.write(f"{label_idx} {x_center:.6f} {y_center:.6f} {w:.6f} {h:.6f}\n")


def generate_path_lists(params):
    train_list_path = os.path.join(params.base_dir, 'train_images.txt')
    all_paths = []

    for pos_dir in params.pos_dir_names:
        folder_path = os.path.join(params.train_dir, pos_dir)
        for img_file in os.listdir(folder_path):
            if img_file.endswith('.jpg'):
                all_paths.append(os.path.abspath(os.path.join(folder_path, img_file)))

    with open(train_list_path, 'w') as f:
        f.writelines([p + '\n' for p in all_paths])
    return train_list_path


class FaceDataset(Dataset):
    def __init__(self, params: Parameters, transform=None):
        self.params = params
        self.transform = transform
        self.label_dir = os.path.join(params.train_dir, 'yolo_labels')  #
        self.image_paths = []

        # Collect all image paths across character folders
        for pos_dir in params.pos_dir_names:  #
            folder_path = os.path.join(params.train_dir, pos_dir)  #
            if os.path.exists(folder_path):
                for img_name in os.listdir(folder_path):
                    if img_name.endswith('.jpg'):
                        self.image_paths.append(os.path.join(folder_path, img_name))

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        img_path = self.image_paths[idx]
        img_name = os.path.basename(img_path)

        # 1. Load Image
        image = cv2.imread(img_path)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)  #

        # 2. Load corresponding YOLO label
        txt_name = os.path.splitext(img_name)[0] + '.txt'
        label_path = os.path.join(self.label_dir, txt_name)

        labels = []
        if os.path.exists(label_path):
            with open(label_path, 'r') as f:
                for line in f:
                    # Format: class_id, x_c, y_c, w, h
                    labels.append(list(map(float, line.strip().split())))

        labels = np.array(labels)
        if self.transform:
            image = self.transform(image)  #

        return image, labels

def create_yolo_train_list(params: Parameters):
    train_file = os.path.join(params.base_dir, 'train_list.txt')
    with open(train_file, 'w') as f:
        for char_dir in params.pos_dir_names: #
            full_path = os.path.join(params.train_dir, char_dir) #
            for img in os.listdir(full_path):
                if img.endswith('.jpg'):
                    f.write(os.path.abspath(os.path.join(full_path, img)) + '\n')
    return train_file


def train_scooby_detector(params: Parameters):
    model = YOLO('yolov5n.pt')
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print(device)
    current_dir = os.path.dirname(__file__)
    yaml_path = os.path.abspath(os.path.join(current_dir, '..', 'yolo_setup', 'data.yaml'))
    if not os.path.exists(yaml_path):
        raise FileNotFoundError(f"YAML file not found at {yaml_path}")
    results = model.train(
        data=yaml_path,
        epochs=5,
        imgsz=300,
        batch=8,
        amp=False,
        device=device,
        # plots=True,
        # cache = False,
        save = True,
        save_period = -1,
        exist_ok = True,
        workers=0
    )

    return model


def get_yolo_predictions(model, params: Parameters):
    all_detections = []
    all_scores = []
    all_file_names = []
    all_labels = []

    if is_test:
        val_base_path = os.path.join(params.dir_test_examples, 'testare')
    else:
        val_base_path = os.path.join(params.dir_validation, 'validare')

    image_paths = []
    for char_dir in params.pos_dir_names:
        folder = os.path.join(val_base_path, char_dir)
        if os.path.exists(folder):
            for img_name in os.listdir(folder):
                if img_name.endswith('.jpg'):
                    image_paths.append(os.path.join(folder, img_name))

    if not image_paths:
        image_paths = [os.path.join(val_base_path, f) for f in os.listdir(val_base_path) if f.endswith('.jpg')]

    print(f"Processing {len(image_paths)} validation images...")

    for img_path in image_paths:
        results = model.predict(source=img_path, conf=0.25, imgsz=300, verbose=False)

        img_name = os.path.basename(img_path)

        for result in results:
            boxes = result.boxes.xyxy.cpu().numpy()
            scores = result.boxes.conf.cpu().numpy()
            classes = result.boxes.cls.cpu().numpy().astype(int)

            for i in range(len(boxes)):
                all_detections.append(boxes[i])
                all_scores.append(scores[i])
                all_file_names.append(img_name)
                char_name = params.pos_dir_names[classes[i]]
                all_labels.append(char_name)

    return np.array(all_detections), np.array(all_scores), np.array(all_file_names), np.array(all_labels)


def prepare_yolo_validation(params: Parameters):
    val_images_path = os.path.join(params.dir_validation, 'validare')
    val_labels_path = os.path.join(params.dir_validation, 'labels')

    if not os.path.exists(val_labels_path):
        os.makedirs(val_labels_path)
        print(f"Created validation labels directory: {val_labels_path}")

    all_val_annots = {}

    for char_name in params.pos_dir_names:
        file_name = f'task2_{char_name}_gt_validare.txt'
        annot_path = os.path.join(params.dir_validation, file_name)

        if not os.path.exists(annot_path):
            print(f"Warning: Validation file {annot_path} not found.")
            continue

        with open(annot_path, 'r') as f:
            for line in f:
                content = line.strip().split(' ')
                # print(content)
                if len(content) < 5: continue

                img_name = content[0]
                bbox = list(map(int, content[1:5]))
                label_name = char_name

                if label_name == 'unknown':
                    continue

                label_idx = params.pos_dir_names.index(label_name)

                if img_name not in all_val_annots:
                    all_val_annots[img_name] = []
                all_val_annots[img_name].append((label_idx, bbox))

    for img_name, annotations in all_val_annots.items():
        img_full_path = os.path.join(val_images_path, img_name)
        # print(img_full_path)
        img = cv2.imread(img_full_path)
        if img is None: continue
        h, w = img.shape[:2]

        txt_name = os.path.splitext(img_name)[0] + '.txt'
        write_path = os.path.join(val_labels_path, txt_name)

        with open(write_path, 'w') as f_out:
            for label_idx, bbox in annotations:
                # Convert to normalized YOLO format
                x_center = (bbox[0] + bbox[2]) / (2 * w)
                y_center = (bbox[1] + bbox[3]) / (2 * h)
                width = (bbox[2] - bbox[0]) / w
                height = (bbox[3] - bbox[1]) / h
                f_out.write(f"{label_idx} {x_center:.6f} {y_center:.6f} {width:.6f} {height:.6f}\n")

def run_yolo():
    params: Parameters = Parameters()

    # print("Generating YOLO labels...")
    # prepare_data(params)
    # prepare_yolo_validation(params)
    # print("Generating training image list...")
    # train_list_file = create_yolo_train_list(params)
    # print(f"Train list created at: {train_list_file}")
    # print("Starting YOLO training...")
    # model = train_scooby_detector(params)

    # model.export(format='onnx')
    model_path = os.path.join(os.path.dirname(__file__), '../../runs/detect/train/weights/best.pt')
    print(model_path)
    if os.path.exists(model_path):
        model = YOLO(model_path)
    else:
        raise FileNotFoundError('YOLO Model not found')

    detections, scores, file_names, labels = get_yolo_predictions(model, params)
    sol_path = os.path.join(params.dir_save_files, 'yolo_solution', 'task1')
    if not os.path.exists(sol_path):
        os.makedirs(sol_path)
    det_path = os.path.join(sol_path, 'detections_all_faces.npy')
    files_path = os.path.join(sol_path, 'file_names_all_faces.npy')
    scores_path = os.path.join(sol_path, 'scores_all_faces.npy')
    np.save(det_path, detections)
    np.save(files_path, file_names)
    np.save(scores_path, scores)

    sol_path2 = os.path.join(params.dir_save_files, 'yolo_solution', 'task2')
    if not os.path.exists(sol_path2):
        os.makedirs(sol_path2)
    for char_name in params.pos_dir_names:
        idx = np.where(labels == char_name)[0]
        char_detections = detections[idx]
        char_scores = scores[idx]
        char_file_names = file_names[idx]
        np.save(os.path.join(sol_path2, f'detections_{char_name}.npy'), char_detections)
        np.save(os.path.join(sol_path2, f'file_names_{char_name}.npy'), char_file_names)
        np.save(os.path.join(sol_path2, f'scores_{char_name}.npy'), char_scores)


    detector = FacialDetector(params)
    detector.eval_detections(detections, scores, file_names)


if __name__ == '__main__':
    run_yolo()
