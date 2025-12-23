import numpy as np
import cv2
import networkx as nx
from networkx.algorithms.shortest_paths.generic import has_path
from itertools import combinations
from scipy.interpolate import splprep, splev, interp1d
from skimage.morphology import skeletonize, remove_small_objects
from scipy.ndimage import binary_fill_holes


def interpolate_centerline(points, num_points=100):
    if len(points) < 2:
        return points
    x, y = zip(*points)
    dist = np.cumsum(np.sqrt(np.diff(x) ** 2 + np.diff(y) ** 2))
    dist = np.insert(dist, 0, 0)
    if dist[-1] == 0:
        return [points[0]] * num_points
    uniform_dist = np.linspace(0, dist[-1], num_points)
    interp_x = interp1d(dist, x, kind='linear')(uniform_dist)
    interp_y = interp1d(dist, y, kind='linear')(uniform_dist)
    return list(zip(interp_x, interp_y))


def smooth_centerline(points):
    if len(points) < 4:
        return points
    try:
        x, y = zip(*points)
        tck, u = splprep([x, y], s=len(points) * 0.5, k=min(3, len(points) - 1))
        new_u = np.linspace(0, 1, len(points))
        smoothed_x, smoothed_y = splev(new_u, tck)
        return list(zip(smoothed_x, smoothed_y))
    except:
        return points


def order_skeleton_points(skeleton_img, points):
    if len(points) == 0:
        return []

    G = nx.Graph()
    point_tuples = [(int(p[0]), int(p[1])) for p in points]
    G.add_nodes_from(point_tuples)

    for y, x in point_tuples:
        for dy, dx in [(-1, -1), (-1, 0), (-1, 1), (0, -1), (0, 1), (1, -1), (1, 0), (1, 1)]:
            n_y, n_x = y + dy, x + dx
            if (0 <= n_y < skeleton_img.shape[0]) and (0 <= n_x < skeleton_img.shape[1]):
                if skeleton_img[n_y, n_x] > 0:
                    G.add_edge((y, x), (n_y, n_x))

    endpoints = [node for node in G if G.degree(node) == 1]

    if len(endpoints) < 2:
        if len(G) == 0 or not nx.is_connected(G):
            return []
        path_lengths = dict(nx.all_pairs_shortest_path_length(G))
        max_dist = 0
        start, end = None, None
        for s in path_lengths:
            for t in path_lengths[s]:
                if path_lengths[s][t] > max_dist:
                    max_dist = path_lengths[s][t]
                    start, end = s, t
        if start and end:
            path = nx.shortest_path(G, start, end)
        else:
            path = list(G.nodes)
    else:
        max_len = 0
        max_path = []
        for s, t in combinations(endpoints, 2):
            if has_path(G, s, t):
                path = nx.shortest_path(G, s, t)
                if len(path) > max_len:
                    max_len = len(path)
                    max_path = path
        path = max_path

    if not path:
        return [(int(points[0][1]), int(points[0][0]))]

    ordered_points = [(x, y) for y, x in path]
    return ordered_points


def extract_skeleton(mask, num_interpolate_points=100):
    if mask.max() > 1:
        mask = (mask > 127).astype(np.uint8)
    kernel = np.ones((3, 3), np.uint8)
    mask_clean = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)
    mask_clean = remove_small_objects(mask_clean.astype(bool), min_size=50).astype(np.uint8)
    mask_filled = binary_fill_holes(mask_clean).astype(np.uint8)

    skeleton = skeletonize(mask_filled > 0).astype(np.uint8)
    points = np.column_stack(np.where(skeleton > 0))

    if len(points) == 0:
        return skeleton, []

    centerline_points = order_skeleton_points(skeleton, points)
    if len(centerline_points) > 3:
        centerline_points = smooth_centerline(centerline_points)
    if len(centerline_points) > 1:
        centerline_points = interpolate_centerline(centerline_points, num_interpolate_points)

    return skeleton, centerline_points


def calculate_curvature(points):
    if len(points) < 3:
        return 0.0
    curvatures = []
    for i in range(1, len(points) - 1):
        p1 = np.array(points[i - 1])
        p2 = np.array(points[i])
        p3 = np.array(points[i + 1])
        v1 = p2 - p1
        v2 = p3 - p2
        cross = np.cross(v1, v2)
        norm_v1 = np.linalg.norm(v1)
        norm_v2 = np.linalg.norm(v2)
        curvature = np.abs(cross) / (norm_v1 * norm_v2) if norm_v1 > 0 and norm_v2 > 0 else 0
        curvatures.append(curvature)
    return np.mean(curvatures) if curvatures else 0.0


def calculate_segment_curvature(points, start_ratio, end_ratio):
    if len(points) < 3:
        return 0.0
    start_idx = int(len(points) * start_ratio)
    end_idx = int(len(points) * end_ratio)
    segment_points = points[start_idx:end_idx]
    if len(segment_points) < 3:
        return 0.0
    return calculate_curvature(segment_points)


def calculate_asymmetry(mask, centerline_points):
    if len(centerline_points) <= 10:
        return 0.0

    try:
        h, w = mask.shape
        asymmetry_values = []
        sample_step = max(1, len(centerline_points) // 20)

        start_idx = int(len(centerline_points) * 0.2)
        end_idx = int(len(centerline_points) * 0.8)

        for i in range(start_idx, end_idx, sample_step):
            center_pt = np.array(centerline_points[i], dtype=np.float32)

            idx_prev = max(0, i - 5)
            idx_next = min(len(centerline_points) - 1, i + 5)
            p_prev = np.array(centerline_points[idx_prev])
            p_next = np.array(centerline_points[idx_next])

            tangent = p_next - p_prev
            tangent_norm = np.linalg.norm(tangent)
            if tangent_norm < 1e-3: continue

            normal = np.array([-tangent[1], tangent[0]]) / tangent_norm

            d_left, d_right = 0, 0
            max_dist = 50

            for r in range(1, max_dist):
                test_pt = center_pt + normal * r
                tx, ty = int(test_pt[0]), int(test_pt[1])
                if tx < 0 or tx >= w or ty < 0 or ty >= h or mask[ty, tx] == 0:
                    d_left = r
                    break

            for r in range(1, max_dist):
                test_pt = center_pt - normal * r
                tx, ty = int(test_pt[0]), int(test_pt[1])
                if tx < 0 or tx >= w or ty < 0 or ty >= h or mask[ty, tx] == 0:
                    d_right = r
                    break

            if d_left > 0 and d_right > 0:
                asymmetry_values.append(abs(d_left - d_right))

        if asymmetry_values:
            return float(np.std(asymmetry_values))
        return 0.0
    except:
        return 0.0