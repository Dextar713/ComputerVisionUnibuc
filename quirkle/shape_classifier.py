import cv2 as cv
import numpy as np
from cv2 import Mat


def resample_contour(cnt: Mat | np.ndarray, n_points:int=200) -> Mat | np.ndarray:
    pts = cnt.reshape(-1, 2).astype(np.float32)

    diff = np.diff(pts, axis=0)
    dist = np.sqrt((diff ** 2).sum(axis=1))
    cumulative = np.concatenate([[0], np.cumsum(dist)])

    total_length = cumulative[-1]
    new_samples = np.linspace(0, total_length, n_points)

    new_pts = np.zeros((n_points, 2), dtype=np.float32)
    new_pts[:, 0] = np.interp(new_samples, cumulative, pts[:, 0])
    new_pts[:, 1] = np.interp(new_samples, cumulative, pts[:, 1])

    return new_pts.reshape(-1, 1, 2)


def scale_contour(cnt: Mat | np.ndarray) -> Mat | np.ndarray:
    pts = cnt.reshape(-1,2)
    pts = pts / np.sqrt((pts**2).sum())
    return pts.reshape(-1,1,2)

def deskew_contour(cnt: Mat | np.ndarray) -> Mat | np.ndarray:
    pts = cnt.reshape(-1,2)
    mean = np.mean(pts, axis=0)
    pts_centered = pts - mean

    cov = np.cov(pts_centered.T)
    eigvals, eigvecs = np.linalg.eig(cov)

    principal_dir = eigvecs[:, np.argmax(eigvals)]
    angle = np.arctan2(principal_dir[1], principal_dir[0])

    M = np.array([[ np.cos(-angle), -np.sin(-angle)],
                  [ np.sin(-angle),  np.cos(-angle)]], dtype=np.float32)

    rotated = pts_centered @ M.T
    return rotated.reshape(-1,1,2)


def normalize(cnt: Mat | np.ndarray) -> Mat | np.ndarray:
    cnt = resample_contour(cnt, 200)
    #cnt = deskew_contour(cnt)
    cnt = scale_contour(cnt)
    return cnt

def get_solidity(contour: np.ndarray) -> float:
    area = cv.contourArea(contour)
    hull = cv.convexHull(contour)
    hull_area = cv.contourArea(hull)
    if hull_area == 0: return 0
    return float(area) / hull_area


def get_orientation_class(contour: np.ndarray) -> str:
    # Fit a rotated rectangle
    rect = cv.minAreaRect(contour)
    angle = rect[-1]

    if angle < -45:
        angle += 90

    angle_tolerance = 20

    if abs(angle) < angle_tolerance or abs(angle - 90) < angle_tolerance:
        return 'square'
    else:
        return 'rhombus'


def get_convexity_defects_count(contour: np.ndarray) -> int:
    hull = cv.convexHull(contour, returnPoints=False)
    if hull is None or len(hull) < 3 or len(contour) < 3:
        return 0

    try:
        defects = cv.convexityDefects(contour, hull)
    except cv.error:
        return 0

    if defects is None: return 0

    count = 0
    x, y, w, h = cv.boundingRect(contour)
    diagonal = np.sqrt(w ** 2 + h ** 2)
    min_depth = diagonal * 0.10

    for i in range(defects.shape[0]):
        s, e, f, d = defects[i, 0]
        depth = d / 256.0
        if depth > min_depth:
            count += 1

    return count


def classify_concave_advanced(contour: np.ndarray) -> str:
    hull = cv.convexHull(contour, returnPoints=False)
    defects = cv.convexityDefects(contour, hull)
    defects_count = defects.shape[0]
    #print('defects count:', defects_count)

    if defects_count >= 5:
        return '7-star'

    elif 3 <= defects_count <= 5:
        far_pts = []
        for i in range(defects_count):
            start, end, far, depth = defects[i, 0]
            far_pts.append(contour[far][0])
        far_pts = np.array(far_pts, dtype=np.float32)
        M = cv.moments(contour)
        cx = M["m10"] / M["m00"]
        cy = M["m01"] / M["m00"]
        center = np.array([cx, cy])

        rel = far_pts - center
        angles = np.arctan2(rel[:, 1], rel[:, 0])
        angles = np.degrees(angles)
        angles = (angles + 360) % 360
        angles.sort()

        angle_tolerance = 20
        angles_mod90 = angles % 90
        mean_angle = np.mean(angles) % 90
        #print(mean_angle)
        near_90_multiples = np.all(angles_mod90 < angle_tolerance)
        near_45_multiples = np.all(np.abs(angles_mod90 - 45) < angle_tolerance)

        if near_90_multiples and not near_45_multiples:
            return "4-star"
        elif near_45_multiples and not near_90_multiples:
            return "plus"
        else:
            mean_angle = np.mean(angles) % 90
            #print(mean_angle)
            #print(mean_angle)
            if abs(mean_angle - 45) < angle_tolerance:
                return "4-star"
            else:
                return "plus"

    else:
        area = cv.contourArea(contour)
        perimeter = cv.arcLength(contour, True)
        if perimeter == 0: return 'unknown'

        circularity = 4 * np.pi * (area / (perimeter * perimeter))

        if circularity > 0.4:
            return '7-star'
        else:
            return 'plus'

def classify_hierarchical(contour: np.ndarray) -> str:
    solidity = get_solidity(contour)

    if solidity > 0.90:
        epsilon = 0.035 * cv.arcLength(contour, True)
        approx = cv.approxPolyDP(contour, epsilon, True)

        if len(approx) > 5:
            return 'circle'

        else:
            return get_orientation_class(contour)

    else:
        best_match = classify_concave_advanced(contour)
        return best_match


def classify_contours(contours: list[np.ndarray], centroids: list) -> list[str]:
    forms = []
    for i, contour in enumerate(contours):
        if cv.contourArea(contour) < 50:
            forms.append("noise")
            continue
        norm_contour = contour
        # norm_contour = normalize(contour)
        norm_contour = cv.approxPolyDP(norm_contour, 0.02 * cv.arcLength(norm_contour, True), True)
        # visualize_reference_contours({f'Unknown {i}': norm_contour})
        best_match = classify_hierarchical(norm_contour)

        # np.savetxt(f'form_templates/contours_{best_match}.txt', contour.squeeze(), delimiter=',', fmt='%f')
        forms.append(best_match)
        #solidity = get_solidity(contour)

        #print(f"{i:<5} {solidity:.3f}      {len(norm_contour):<10} {best_match}")
    #print('Centroids:', centroids)
    #print(f"Forms: {forms}")
    return forms