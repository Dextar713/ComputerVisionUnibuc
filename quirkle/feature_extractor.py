import numpy as np
import cv2 as cv
import os
from cv2 import Mat
import warnings

from quirkle.position_locator import QwirkleGame, decode_config, encode_config
from quirkle.qwirkle_base import colors, Piece
from quirkle.shape_classifier import classify_contours

warnings.filterwarnings('ignore')

# Source - https://stackoverflow.com/a/66269158
# Posted by Ali Hashemian
# Retrieved 2025-11-18, License - CC BY-SA 4.0

color_dict_HSV = {'black': [[180, 255, 60], [0, 0, 0]],
              'white': [[180, 24, 255], [0, 0, 187]],
              'red1': [[180, 255, 255], [159, 50, 61]],
              'red2': [[9, 255, 255], [0, 50, 70]],
              'green': [[89, 255, 255], [36, 50, 70]],
              'blue': [[128, 255, 255], [90, 50, 70]],
              'yellow': [[35, 255, 255], [25, 50, 70]],
              'purple': [[158, 255, 255], [129, 50, 70]],
              'orange': [[19, 255, 255], [0, 50, 70]],
              'gray': [[180, 21, 250], [0, 0, 70]]}

NUM_CELLS = 8
# mode = 'train'
mode = 'fake_test'
IMG_NAME = '1_07.jpg'
#'red2': [[10, 255, 255], [0, 50, 65]],

def read_image(img_name: str) -> np.ndarray:
    cur_dir = os.getcwd()
    if mode == 'fake_test':
        image_path = os.path.join(cur_dir, 'evaluare', 'fake_test', img_name)
    elif mode == 'train':
        image_path = os.path.join(cur_dir, 'antrenare', img_name)
    else:
        image_path = os.path.join(cur_dir, 'evaluare', 'fake_test', img_name)
    # print(image_path)
    img = cv.imread(image_path, cv.IMREAD_COLOR)
    img = cv.resize(img, dsize=(0, 0), fx=0.2, fy=0.2, interpolation=cv.INTER_AREA)
    return img

def detect_table_corners(img: Mat | np.ndarray) -> tuple[Mat | np.ndarray, Mat | np.ndarray]:
    gray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)

    corners = cv.goodFeaturesToTrack(gray, maxCorners=0, qualityLevel=0.07, minDistance=10)
    corners = np.int_(corners)

    coords = np.array([i.ravel() for i in corners])
    xy_sum = np.sum(coords, axis=1)
    xy_diff = coords[:, 0] - coords[:, 1]
    top_left_idx = np.argmin(xy_sum)
    bottom_right_idx = np.argmax(xy_sum)
    top_right_idx = np.argmax(xy_diff)
    bottom_left_idx = np.argmin(xy_diff)
    table_corners_idx = np.array([top_left_idx, top_right_idx, bottom_left_idx, bottom_right_idx])
    table_corners: np.ndarray = coords[table_corners_idx]
    for corner_idx in table_corners_idx:
        x, y = coords[corner_idx]
        cv.circle(img, (x, y), 5, (0, 255, 0), -1)
    return img, table_corners

def extract_table(img: Mat | np.ndarray, table_corners: Mat | np.ndarray) -> np.ndarray:
    top_left, top_right, bottom_left, bottom_right = table_corners
    H = np.mean([bottom_left[1] - top_left[1], bottom_right[1] - top_right[1]]).astype(int)
    W = np.mean([bottom_right[0] - bottom_left[0], top_right[0] - top_left[0]]).astype(int)

    extra_table_padding = 10
    H += extra_table_padding * 2
    W += extra_table_padding * 2
    top_left -= extra_table_padding
    bottom_right += extra_table_padding
    bottom_left[0] -= extra_table_padding
    bottom_left[1] += extra_table_padding
    top_right[0] += extra_table_padding
    top_right[1] -= extra_table_padding
    table_corners = np.vstack([top_left, top_right, bottom_left, bottom_right])

    new_coords = np.array([(0, 0), (W, 0), (0, H), (W, H)])
    M = cv.getPerspectiveTransform(table_corners.astype(np.float32), new_coords.astype(np.float32))
    game_table_img = cv.warpPerspective(img, M, (W, H))
    cv.imwrite('visuals/table_image.jpg', game_table_img)
    return game_table_img


def extract_pieces_by_black_border(img: Mat | np.ndarray) -> np.ndarray:
    hsv = cv.cvtColor(img, cv.COLOR_BGR2HSV)
    lower_black = np.array([0, 0, 0])
    upper_black = np.array([180, 255, 68])

    border_mask = cv.inRange(hsv, lower_black, upper_black)

    kernel = np.ones((5,5), np.uint8)
    closed = cv.morphologyEx(border_mask, cv.MORPH_CLOSE, kernel, iterations=2)
    filled = cv.morphologyEx(closed, cv.MORPH_CLOSE, kernel, iterations=6)
    #show_image(filled)

    contours, _ = cv.findContours(filled, cv.RETR_EXTERNAL, cv.CHAIN_APPROX_SIMPLE)

    result = img.copy()
    cv.drawContours(result, contours, -1, (0,255,0), 3)
    cv.imwrite('visuals/PiecesContours.jpg', result)

    mask = np.zeros(img.shape[:2], dtype=np.uint8)
    for c in contours:
        cv.drawContours(mask, [c], -1, (255,), thickness=cv.FILLED)

    extracted = cv.bitwise_and(img, img, mask=filled)
    cv.imwrite('visuals/PiecesOnly.jpg', extracted)
    return extracted

def extract_pieces_by_color(img: Mat | np.ndarray, color: str = 'green') -> Mat | np.ndarray:
    hsv_img = cv.cvtColor(img, cv.COLOR_BGR2HSV)
    result = hsv_img.copy()

    lowerLimit = np.array(color_dict_HSV[color][1])
    upperLimit = np.array(color_dict_HSV[color][0])

    mask = cv.inRange(hsv_img, lowerLimit, upperLimit)
    mask = cv.medianBlur(mask, 3)
    return mask

def extract_contours(binary_img: Mat | np.ndarray) -> Mat | tuple[np.ndarray]:
    #edges = cv.Canny(img, 100, 200)
    #gray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
    #_, binary_img = cv.threshold(gray, 10, 255, cv.THRESH_BINARY)
    contours, hierarchy = cv.findContours(binary_img, cv.RETR_EXTERNAL, cv.CHAIN_APPROX_SIMPLE)
    max_contour = max(contours, key=lambda item: cv.contourArea(item))
    max_area = cv.contourArea(max_contour)
    cutoff_area = max_area / 4
    filtered_contours = []
    for contour in contours:
        if cv.contourArea(contour) > cutoff_area:
            filtered_contours.append(contour)

    centroids = []
    blank_img = np.zeros_like(binary_img)

    for contour in filtered_contours:
        M = cv.moments(contour)
        if M['m00'] != 0:
            cx = int(M['m10'] / M['m00'])
            cy = int(M['m01'] / M['m00'])
            cv.drawContours(blank_img, [contour], -1, (0, 255, 0), 2)
            cv.circle(blank_img, (cx, cy), 7, (255, 0, 255), -1)
            centroids.append([cx, cy])

    cv.imwrite('visuals/centers_contours.jpg', blank_img)
    cv.imwrite('visuals/BinaryPiecesByColor.jpg', binary_img)
    return filtered_contours, centroids


def show_image(img: Mat | np.ndarray) -> None:
    cv.imshow('image', img)
    cv.waitKey(0)
    cv.destroyAllWindows()

def start_game(pieces_image: Mat | np.ndarray) -> QwirkleGame:
    game_pieces: list[Piece] = []
    for color in colors:
        result_image = extract_pieces_by_color(pieces_image, color)
        contours, centers = extract_contours(result_image)
        contour_forms = classify_contours(contours, centers)
        for i in range(len(contour_forms)):
            cur_piece = Piece(color, contour_forms[i], centers[i])
            game_pieces.append(cur_piece)
    game = QwirkleGame(pieces_image.shape[:2])
    game.place_pieces(game_pieces)
    return game

def add_bonus_cells(game: QwirkleGame, table_image: np.ndarray) -> None:
    numbers_contours = extract_numbers_by_orange_border(table_image[10:-10, 10:-10])
    numbers = classify_numbers(numbers_contours)
    numbers_centroids = get_contours_centroids(numbers_contours)
    game.add_bonus_numbers(numbers, numbers_centroids)

def get_table_image(img_name: str) -> np.ndarray:
    image = read_image(img_name)
    img_with_corners, corners = detect_table_corners(image.copy())
    table_image = extract_table(image, corners)
    return table_image

def encode_game(cur_img_name: str, start_img_name: str) -> tuple[QwirkleGame, np.ndarray]:
    cur_table_image = get_table_image(cur_img_name)
    pieces_image = extract_pieces_by_black_border(cur_table_image)
    game = start_game(pieces_image)
    start_table_image = get_table_image(start_img_name)
    add_bonus_cells(game, start_table_image)
    encoded_config = game.get_encoded_config()
    return game, encoded_config

def write_config(decoded_conf: list[list[str]], test_set_num:int, test_num:int) -> None:
    if mode == 'train':
        file = open(f'train_output/{test_set_num}_{test_num:02d}.txt', 'w+')
    elif mode == 'fake_test':
        file = open(f'fake_test_output/{test_set_num}_{test_num:02d}.txt', 'w+')
    else:
        file = open(f'fake_test_output/{test_set_num}_{test_num:02d}.txt', 'w+')
    for i in range(len(decoded_conf)):
        for j in range(len(decoded_conf[i])):
            if decoded_conf[i][j] != '':
                output_label = f'{i+1}{chr(ord('A')+j)} {decoded_conf[i][j]}\n'
                file.write(output_label)
                #print(output_label)
    file.close()

def write_score(score:int, test_set_num:int, test_num:int) -> None:
    if mode == 'train':
        file = open(f'train_output/{test_set_num}_{test_num:02d}.txt', 'a')
    elif mode == 'fake_test':
        file = open(f'fake_test_output/{test_set_num}_{test_num:02d}.txt', 'a')
    else:
        file = open(f'fake_test_output/{test_set_num}_{test_num:02d}.txt', 'a')
    file.write(str(score))
    file.close()

def extract_numbers_by_orange_border(img: Mat | np.ndarray) -> list[np.ndarray]:
    hsv = cv.cvtColor(img, cv.COLOR_BGR2HSV)
    color = 'orange'
    lower_orange = np.array(color_dict_HSV[color][1])
    upper_orange = np.array(color_dict_HSV[color][0])

    mask = cv.inRange(hsv, lower_orange, upper_orange)
    kernel = np.ones((5, 5), np.uint8)
    closed = cv.morphologyEx(mask, cv.MORPH_CLOSE, kernel, iterations=2)
    mask = cv.morphologyEx(closed, cv.MORPH_CLOSE, kernel, iterations=6)
    result = cv.bitwise_and(img, img, mask=mask)

    color = 'gray'
    upperWhite = np.array(color_dict_HSV[color][0])
    lowerWhite = np.array(color_dict_HSV[color][1])
    hsv2 = cv.cvtColor(result, cv.COLOR_BGR2HSV)
    mask2 = cv.inRange(hsv2, lowerWhite, upperWhite)
    kernel = np.ones((5, 5), np.uint8)
    mask2 = cv.morphologyEx(mask2, cv.MORPH_CLOSE, kernel, iterations=5)
    result = cv.bitwise_and(result, result, mask=mask2)
    hsv2 = cv.cvtColor(result, cv.COLOR_BGR2HSV)
    mask2 = cv.inRange(hsv2, lower_orange, upper_orange)
    result = cv.bitwise_and(result, result, mask=mask2)
    cv.imwrite('visuals/NumbersOnly.jpg', result)

    contours, _ = cv.findContours(mask2, cv.RETR_EXTERNAL, cv.CHAIN_APPROX_SIMPLE)
    filtered_contours = []
    max_contour = max(contours, key=lambda item: cv.contourArea(item))
    max_area = cv.contourArea(max_contour)
    cutoff_area = max_area / 3
    for contour in contours:
        if cv.contourArea(contour) > cutoff_area:
            filtered_contours.append(contour)
    cv.drawContours(result, filtered_contours, -1, (0, 255, 0), 5)
    cv.imwrite('visuals/NumbersContours.jpg', result)
    return filtered_contours

def get_contours_centroids(contours: list[np.ndarray]) -> list[tuple[int, int]]:
    centroids = []
    for contour in contours:
        M = cv.moments(contour)
        if M['m00'] != 0:
            cx = int(M['m10'] / M['m00'])
            cy = int(M['m01'] / M['m00'])
            centroids.append((cx, cy))
    return centroids


def classify_numbers(contours: list[np.ndarray]) -> list[int]:
    contour_boxes = [cv.minAreaRect(cnt) for cnt in contours]
    boxes_side_ratios = []
    for i, box in enumerate(contour_boxes):
        (x, y), (w, h), angle = box
        max_side, min_side = max(w, h), min(w, h)
        side_ratio = max_side / min_side
        boxes_side_ratios.append(side_ratio)

    max_ratio = max(boxes_side_ratios)
    min_ratio = min(boxes_side_ratios)
    numbers = []
    for i in range(len(contours)):
        #print(contour_boxes[i][0])
        side_ratio = boxes_side_ratios[i]
        if abs(side_ratio - max_ratio) < abs(side_ratio - min_ratio):
            numbers.append(1)
        else:
            numbers.append(2)
        #print(numbers[i])
    return numbers

# def get_full_game(img_name: str) -> QwirkleGame:
#     table_image = get_table_image(img_name)
#     pieces_image = extract_pieces_by_black_border(table_image)
#     game = start_game(pieces_image)
#     add_bonus_cells(game, table_image)


if __name__ == '__main__':
    test_set_num = 1
    test_num = 16
    start_img_name = f'{test_set_num}_{test_num - 1:02d}.jpg'
    cur_img_name = f'{test_set_num}_{test_num:02d}.jpg'
    table_image = get_table_image(cur_img_name)
    pieces_image = extract_pieces_by_black_border(table_image)
    show_image(table_image)
    show_image(pieces_image)
    #game = start_game(pieces_image)
    #game = add_bonus_cells(game, table_image)
        # pieces_image = extract_pieces_by_black_border(table_image)
    # color = 'red1'
    # result_image = extract_pieces_by_color(pieces_image, color)
    # show_image(result_image)
    # contours, centers = extract_contours(result_image)
    # contour_forms = classify_contours(contours, centers)
    # print('centers: ', centers)
    # print('forms: ', contour_forms)




