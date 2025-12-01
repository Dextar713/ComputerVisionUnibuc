# 🎲 Qwirkle Game Computer Vision Project

**Repository:** https://github.com/Dextar713/ComputerVisionUnibuc

---

## 📘 Project Overview

This project aims to automatically classify the state of a **Qwirkle board game** from an input image.  
The system detects the game table, identifies newly placed pieces (both **color** and **shape**), and calculates the **score for the current move** based on official game rules and bonus cells.

---

## 🚀 Key Features & Algorithms

The project uses **OpenCV** for feature extraction and geometric classification — **no template matching**, ensuring **scale- and rotation-invariance**.

---

## 1️⃣ Table & Piece Extraction

### 🟦 Table Detection
- Detects table corners using **`goodFeaturesToTrack`**  
- Applies a **perspective transform** to obtain a normalized, top-down board view.

### 🟩 Piece Segmentation
- Extracts pieces by isolating their **black borders** using morphological operations.  
- Applies **HSV color masking** to classify pieces by color.

---

## 2️⃣ Shape Classification

Shapes are classified using **geometric properties** rather than template matching.

Contours are polygon-approximated and divided into:

### ✅ **Convex Shapes**
- **Square**, **Rhombus**, **Circle**
- Classified using:
  - **Solidity (>0.90)**
  - **Bounding rectangle angle**  
    - Rhombus ≈ 45°  
    - Square ≈ 0°

### ❎ **Concave Shapes**
Classification uses **convexity defects**:

- **8-Star:** ≥ 5 significant defects  
- **Plus vs. 4-Star:**  
  - Compute angles of defect points around the centroid  
  - **45° rotation indicates a 4-Star**

---

## 3️⃣ Bonus Cells & Scoring

### 🔶 Bonus Extraction
- Extracts orange/red cells (digits **1** and **2**) using HSV masks.

### 🔢 Number Classification
- **'1'** vs **'2'** differentiated by bounding rectangle **aspect ratio**  
  - '1' is more stretched vertically.

### 🧮 Game Logic
- Board state is stored in a **matrix**.
- Score is computed by comparing the **current** vs **previous** board state.
- Detects valid lines and **Qwirkles (6-in-a-row)**.

---

## 📦 Installation

Make sure the following dependencies are installed:

```bash
pip install numpy==2.3.5
pip install opencv-python==4.11.0.86
pip install opencv-contrib-python==4.11.0.86
```

How to Run:

The main entry point for the project is quirkle/run_project.py.

Open quirkle/run_project.py.

Configure the parameters in the run() function which calls run_tests:


```python
run_tests(
    input_img_path="path/to/test_images",   # Relative path to directory with input images 
    output_conf_path="path/to/results",     # Relative path to directory for output .txt files 
    tests_set_cnt=4,                        # Number of test sets 
    tests_cnt=10,                           # Number of tests per test set
    eval_mode=False,                        # Set to False for execution
    input_conf_path=""                      # Can be empty when eval_mode is False 
)
```
Run the script:

```bash
python -m quirkle.run_project
```

Note: The project may generate intermediary images in visuals directory and print messages to console for debugging purposes; these can be ignored.
