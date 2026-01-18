import pickle

import cv2
from skimage.feature import hog

from MovieFaceDetection.code.image_processing import ImageProcessor, get_image_shapes
from MovieFaceDetection.code.Parameters import Parameters, is_test
from MovieFaceDetection.code.FacialDetector import FacialDetector
from MovieFaceDetection.code.Visualize import show_detections_without_ground_truth, show_detections_with_ground_truth
import numpy as np
import os

def final_nms_per_image(detector, detections, scores, file_names, img_shapes):
    final_dets = []
    final_scores = []
    final_names = []

    for img_name in np.unique(file_names):
        idx = np.where(file_names == img_name)[0]

        dets = detections[idx]
        scrs = scores[idx]

        if len(dets) == 0:
            continue

        h, w = img_shapes[img_name]
        keep_dets, keep_scores = detector.non_maximal_suppression(
            dets, scrs, (h, w)
        )

        final_dets.append(keep_dets)
        final_scores.append(keep_scores)
        final_names.extend([img_name] * len(keep_scores))

    return (
        np.vstack(final_dets),
        np.hstack(final_scores),
        np.array(final_names)
    )



def main():
    params: Parameters = Parameters()
    params.dim_window = 64  # exemplele pozitive (fete de oameni cropate) au 36x36 pixeli
    params.dim_hog_cell = 8  # dimensiunea celulei
    params.overlap = 0.3
    params.number_positive_examples = 2000  # numarul exemplelor pozitive
    params.number_negative_examples = 2000  # numarul exemplelor negative

    # 3.5 thresh = 0.629 AP score
    # 64 dim_window, 8 dim_hog_cell, 4.5 thresh, 6 neg samples, 7 padding
    params.threshold = 4.5 # toate ferestrele cu scorul > threshold si maxime locale devin detectii
    params.has_annotations = True

    params.use_hard_mining = False  # (optional)antrenare cu exemple puternic negative
    params.use_flip_images = False  # adauga imaginile cu fete oglindite

    if params.use_flip_images:
        params.number_positive_examples *= 2

    image_processor = ImageProcessor(params)
    facial_detector: FacialDetector = FacialDetector(params)

    #face_windows = [(64, 64), (64, 80), (80, 64), (40, 40), (48, 56)]
    # face_windows = [(64, 64),(56, 64),(64, 80), (40, 72),(48, 56), (32, 40), (32, 32), (24, 24), (96, 96)]
    # face_windows = [(56, 64)]   #40,72 minus  64, 80 - 48,56 - -56,64
    face_windows = [(64, 64), (56,64), (48,56), (24, 24), (96, 96), (96, 104)]
    # 5.0 # 2.5 # 4.5
    all_detections = []
    all_scores = []
    all_file_names = []
    for face_window in face_windows:
        params.dim_face = face_window
        if face_window == (24, 24):
            params.threshold = 2.4
        elif face_window[0] >= 96:
            params.threshold = 5.5
        else:
            params.threshold = 4.5
        # Pasii 1+2+3. Incarcam exemplele pozitive (cropate) si exemple negative generate
        # verificam daca sunt deja existente
        positive_features, negative_features = None, None
        positive_features_path = os.path.join(params.dir_save_files, 'descriptors', f'pos_descriptors_{face_window[0]}_{face_window[1]}.npy')
        if os.path.exists(positive_features_path):
            positive_features = np.load(positive_features_path)
            print('Am incarcat descriptorii pentru exemplele pozitive')

        # exemple negative
        negative_features_path = os.path.join(params.dir_save_files, 'descriptors', f'neg_descriptors_{face_window[0]}_{face_window[1]}.npy')
        if os.path.exists(negative_features_path):
            negative_features = np.load(negative_features_path)
            print('Am incarcat descriptorii pentru exemplele negative')

        if positive_features is None:
            if negative_features is None:
                positive_features, negative_features, _= image_processor.generate_train_data(with_negative=True)
            else:
                positive_features, _, _= image_processor.generate_train_data(with_negative=False)
            image_processor.save_descriptors(positive_features, negative_features)

        print(positive_features.shape)
        print(negative_features.shape)

        # Pasul 4. Invatam clasificatorul liniar
        training_examples = np.concatenate((np.squeeze(positive_features), np.squeeze(negative_features)), axis=0)
        train_labels = np.concatenate((np.ones(positive_features.shape[0]), np.zeros(negative_features.shape[0])))
        facial_detector.train_classifier(training_examples, train_labels, classify_flag=False)

        # Pasul 5. (optional) Antrenare cu exemple puternic negative (detectii cu scor >0 din cele 274 de imagini negative)
        # Daca implementati acest pas ar trebui sa modificati functia FacialDetector.run()
        # astfel incat sa va returneze descriptorii detectiilor cu scor > 0 din cele 274 imagini negative
        # completati codul in continuare
        # TODO:  (optional)  completeaza codul in continuare

        detections, scores, file_names = facial_detector.run()
        all_detections.append(detections)
        all_scores.append(scores)
        all_file_names.append(file_names)


    all_detections = np.vstack(all_detections)
    all_scores = np.hstack(all_scores)
    all_file_names = np.hstack(all_file_names)
    if not is_test:
        img_shapes = get_image_shapes(params.dir_validation)
    else:
        img_shapes = get_image_shapes(params.dir_test_examples)
    detections, scores, file_names = final_nms_per_image(
        facial_detector,
        all_detections,
        all_scores,
        all_file_names,
        img_shapes
    )
    # sol_path = os.path.join(params.dir_save_files, 'solution')
    # det_path = os.path.join(sol_path, 'detections_all_faces.npy')
    # files_path = os.path.join(sol_path, 'file_names_all_faces.npy')
    # scores_path = os.path.join(sol_path, 'scores_all_faces.npy')
    # np.save(det_path, detections)
    # np.save(files_path, file_names)
    # np.save(scores_path, scores)

    if params.has_annotations:
        facial_detector.eval_detections(detections, scores, file_names)
        show_detections_with_ground_truth(detections, scores, file_names, params)
    else:
        show_detections_without_ground_truth(detections, scores, file_names, params)

def test_recognition():
    params: Parameters = Parameters()
    image_processor = ImageProcessor(params)
    facial_detector: FacialDetector = FacialDetector(params)
    labels_path = os.path.join(params.dir_save_files, 'pos_labels', 'pos_labels.npy')
    recognition_face_size = (96, 96)
    params.dim_face = recognition_face_size

    if os.path.exists(labels_path):
        pos_labels = np.load(labels_path)
        positive_f, _ = image_processor.load_descriptors(with_color=True)
        if positive_f is None:
            positive_f, negative_f, pos_labels = image_processor.generate_train_data(with_negative=False, with_color=True)
            image_processor.save_descriptors(positive_f, negative_f, with_color=True)

    else:
        positive_f, negative_f, pos_labels = image_processor.generate_train_data(with_negative=False)
        image_processor.save_descriptors(positive_f, negative_f)
        np.save(labels_path, pos_labels)
    # print(pos_labels[:5])
    facial_detector.train_classifier(positive_f, pos_labels, classify_flag=True, with_color=True)
    sol_path = os.path.join(params.dir_save_files, 'solution', 'task1')
    det_path = os.path.join(sol_path, 'detections_all_faces.npy')
    files_path = os.path.join(sol_path, 'file_names_all_faces.npy')
    scores_path = os.path.join(sol_path, 'scores_all_faces.npy')
    detections = np.load(det_path)
    file_names = np.load(files_path)
    scores = np.load(scores_path)
    test_features = []
    img_id = 0
    color = ''
    if params.with_color:
        color = 'color'
    val_desc_path = os.path.join(params.dir_save_files, 'validation_descriptors', f'descriptors_{params.dim_face[0]}_{params.dim_face[1]}{color}.npy')
    if not os.path.exists(val_desc_path):
        for (detection, score, file_name) in zip(detections, scores, file_names):
            if not is_test:
                img_path = os.path.join(params.dir_validation, 'validare', file_name)
            else:
                img_path = os.path.join(params.dir_test_examples, 'testare', file_name)
            img = cv2.imread(str(img_path), cv2.IMREAD_COLOR_BGR)
            cur_face = img[detection[1]:detection[-1], detection[0]:detection[-2]]
            cur_face = cv2.resize(cur_face, recognition_face_size, interpolation=cv2.INTER_AREA)
            # if img_id % 10 == 0:
            #     cv2.imshow('img', cur_face)
            #     cv2.waitKey(0)
            img_id += 1
            gray_face = cv2.cvtColor(cur_face, cv2.COLOR_BGR2GRAY)
            hog_descriptors = hog(gray_face, pixels_per_cell=(params.dim_hog_cell, params.dim_hog_cell),
                                  cells_per_block=(2, 2), feature_vector=True)
            if not params.with_color:
                test_features.append(hog_descriptors)
            else:
                hsv_face = cv2.cvtColor(cur_face, cv2.COLOR_BGR2HSV)
                h_hist = cv2.calcHist([hsv_face], [0], None, [params.hist_size], [0, 180])
                s_hist = cv2.calcHist([hsv_face], [1], None, [params.hist_size], [0, 256])
                v_hist = cv2.calcHist([hsv_face], [2], None, [params.hist_size], [0, 256])
                color_feat = np.concatenate([h_hist, s_hist, v_hist]).flatten()
                color_feat = color_feat / (np.sum(color_feat) + 1e-7)
                hybrid_feat = np.concatenate([hog_descriptors, color_feat])
                test_features.append(hybrid_feat)
        test_features = np.array(test_features)
        np.save(val_desc_path, test_features)
    else:
        test_features = np.load(val_desc_path)
    svm_file_name = os.path.join(params.dir_save_files, 'models', 'best_model_%d_%d_%d' %
                                 (params.dim_hog_cell, params.dim_face[0],
                                  params.dim_face[1]))
    # svm_file_name = svm_file_name + '_class'
    svm_file_name = svm_file_name + '_class_color'
    model = pickle.load(open(svm_file_name, 'rb'))

    scores_matrix = model.decision_function(test_features)
    predicted_labels = np.argmax(scores_matrix, axis=1)
    confidence_scores = np.max(scores_matrix, axis=1)
    print(np.max(confidence_scores), np.min(confidence_scores), np.mean(confidence_scores))
    recognition_threshold = -7.0
    keep_idx = np.where(confidence_scores > recognition_threshold)

    task2_path = os.path.join(params.dir_save_files, 'solution', 'task2')
    if not os.path.exists(task2_path):
        os.makedirs(task2_path)
    # print(predicted_labels[:5])
    for i, character_name in enumerate(params.pos_dir_names):
        cur_idx = np.where(predicted_labels[keep_idx] == i)[0]
        np.save(os.path.join(task2_path, f'detections_{character_name}.npy'), detections[keep_idx][cur_idx])
        np.save(os.path.join(task2_path, f'file_names_{character_name}.npy'), file_names[keep_idx][cur_idx])
        np.save(os.path.join(task2_path, f'scores_{character_name}.npy'), confidence_scores[keep_idx][cur_idx])


if __name__ == '__main__':
    main()
    test_recognition()
