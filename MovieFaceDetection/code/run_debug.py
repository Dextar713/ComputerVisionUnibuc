import numpy as np

from MovieFaceDetection.code.Parameters import Parameters
import os

if __name__ == '__main__':
    params = Parameters()
    for pos_dir_name in params.pos_dir_names:
        file_name = f'{pos_dir_name}_annotations.txt'
        annot_path = os.path.join(params.train_dir, file_name)
        face_bboxes = []
        with open(annot_path, 'r') as f:
            for line in f:
                content = line.strip().split(' ')
                img_name = content[0]
                bbox = list(map(int, content[1:5]))
                face_bboxes.append((bbox[2]-bbox[0], bbox[3]-bbox[1]))
        face_bboxes = np.array(face_bboxes)
        mn_x, mn_y = np.min(face_bboxes[:, 0]), np.min(face_bboxes[:, 1])
        mx_x, mx_y = np.max(face_bboxes[:, 0]), np.max(face_bboxes[:, 1])
        mean_x, mean_y = np.mean(face_bboxes[:, 0]), np.mean(face_bboxes[:, 1])
        print('Max delta x, delta y:', mx_x, mx_y)
        print('Min delta x, delta y:', mn_x, mn_y)
        print('Mean delta x, delta y:', mean_x, mean_y)
