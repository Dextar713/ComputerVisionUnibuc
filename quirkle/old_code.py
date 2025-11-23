# def get_reference_contours() -> dict[str, Mat | tuple[np.ndarray]]:
#     square_contour_array = np.loadtxt('form_templates/contours_square_200.txt', dtype=np.int_, delimiter=',')
#     circle_contour_array = np.loadtxt('form_templates/contours_circle_200.txt', dtype=np.int_, delimiter=',')
#     rhombus_contour_array = np.loadtxt('form_templates/contours_rhombus_200.txt', dtype=np.int_, delimiter=',')
#     plus_contour_array = np.loadtxt('form_templates/contours_plus.txt', dtype=np.int_, delimiter=',')
#     star4_contour_array = np.loadtxt('form_templates/contours_4-star.txt', dtype=np.int_, delimiter=',')
#     star7_contour_array = np.loadtxt('form_templates/contours_7-star.txt', dtype=np.int_, delimiter=',')
#
#     REFERENCE_CONTOURS = {
#         'square': square_contour_array,
#         'circle': circle_contour_array,
#         'rhombus': rhombus_contour_array,
#         'plus': plus_contour_array,
#         '4-star': star4_contour_array,
#         '7-star': star7_contour_array
#     }
#     return REFERENCE_CONTOURS



# def visualize_reference_contours(references: dict[str, np.ndarray],
#                                  canvas_size=(400, 400)) -> None:
#
#     for shape_name, contour in references.items():
#         contour = contour.squeeze()
#         canvas = np.ones((canvas_size[0], canvas_size[1], 3), dtype=np.uint8) * 255
#         # Normalize contour to fit nicely inside the canvas
#         c = contour.copy().astype(np.float32)
#
#         # Shift contour so minimum is at (0,0)
#         min_xy = c.min(axis=0)
#         c -= min_xy
#
#         # Scale contour to ~70% of canvas
#         max_xy = c.max(axis=0)
#         scale = 0.7 * min(canvas_size) / max(max_xy)
#         c *= scale
#
#         # Center contour in canvas
#         offset_x = (canvas_size[1] - c[:, 0].max()) // 2
#         offset_y = (canvas_size[0] - c[:, 1].max()) // 2
#         c[:, 0] += offset_x
#         c[:, 1] += offset_y
#         c = c.reshape(-1, 1, 2).astype(np.int32)
#         cv.drawContours(canvas, [c], -1, (0, 0, 0), 2)
#
#         # Show
#         # cv.imshow(f"Reference: {shape_name}", canvas)
#         cv.imwrite(f"visuals/Reference_{shape_name}_{len(references)}.jpg", canvas)
#
#     cv.waitKey(0)
#     cv.destroyAllWindows()

# def extract_dominant_colors(img: Mat | np.ndarray) -> np.ndarray:
#     img_lab = cv.cvtColor(img, cv.COLOR_BGR2Lab)
#     pixels = img_lab.reshape(-1, 3)
#     kmeans = KMeans(n_clusters=7, random_state=0)
#     kmeans.fit(pixels)
#     lab_colors = kmeans.cluster_centers_.astype(np.uint8)
#     rgb_colors = cv.cvtColor(lab_colors.reshape(1, -1, 3), cv.COLOR_LAB2BGR).reshape(-1, 3)
#     return rgb_colors.astype(np.uint8)
#
# def display_dominant_colors(colors: Mat | np.ndarray) -> None:
#     plt.figure(figsize=(10, 2))
#     for i, color in enumerate(colors):
#         plt.fill_between([i, i+1], 0, 1, color=color/255)
#     plt.axis('off')
#     plt.title('Dominant Colors')
#     plt.show()