import os

is_test = False

class Parameters:
    def __init__(self):
        self.base_dir = os.path.join(os.path.dirname(__file__), '..', 'CAVA-2025-TEMA2')
        self.train_dir = os.path.join(self.base_dir, 'antrenare')
        self.pos_dir_names = ['daphne', 'fred', 'shaggy', 'velma']
        self.dir_neg_examples = os.path.join(self.train_dir, 'exempleNegative')
        self.dir_validation = os.path.join(self.base_dir, 'validare')
        self.dir_test_examples = os.path.join(self.base_dir,'testare')
        self.dir_save_files = os.path.join(self.base_dir, 'saved_files')
        if not os.path.exists(self.dir_save_files):
            os.makedirs(self.dir_save_files)
            print('directory created: {} '.format(self.dir_save_files))
        else:
            print('directory {} exists '.format(self.dir_save_files))

        self.dim_window = 64  # exemplele pozitive (fete de oameni cropate) au 36x36 pixeli
        self.dim_hog_cell = 8  # dimensiunea celulei
        self.dim_descriptor_cell = 64  # dimensiunea descriptorului unei celule
        self.overlap = 0.3
        self.number_positive_examples = 6713  # numarul exemplelor pozitive
        self.number_negative_examples = 10000  # numarul exemplelor negative
        self.has_annotations = False
        self.path_annotations = os.path.join(self.dir_validation, 'task1_gt_validare.txt')
        self.scale_factor_w = 1.05
        self.scale_factor_h = 1.05
        self.threshold = 0
        self.dim_face = (64, 64)   # w, h
        self.with_color = True
        self.hist_size = 16