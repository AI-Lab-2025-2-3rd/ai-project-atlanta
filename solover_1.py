import math
from typing import List
import numpy as np
import pandas as pd
import time

CAPACITY = 25
DATA_CSV = "data.csv"
OUT_CSV = "solution.csv"

# ---------- 0) 데이터 로드 (가정: point_id에 'DEPOT' 존재) ----------
def load_data(path):
    df = pd.read_csv(path)
    depot_row = df[df["point_id"] == "DEPOT"].iloc[0]
    depot_id = depot_row["point_id"]
    depot_xy = np.array([float(depot_row["x"]), float(depot_row["y"])], dtype=float)

    customers = df[df["point_id"] != depot_id].copy().reset_index(drop=True)
    customers["demand"] = customers["demand"].astype(int)

    ids = [depot_id] + customers["point_id"].tolist()
    x = np.array([depot_xy[0]] + customers["x"].astype(float).tolist(), dtype=float)
    y = np.array([depot_xy[1]] + customers["y"].astype(float).tolist(), dtype=float)
    demands = np.array([0] + customers["demand"].tolist(), dtype=int)
    return ids, np.stack([x, y], axis=1), demands

# ---------- 1) 거리 행렬 ----------
def build_dist_matrix(points):
    n = points.shape[0]
    D = np.zeros((n, n), dtype=float)
    for i in range(n):
        for j in range(i + 1, n):
            d = float(np.hypot(points[i,0] - points[j,0], points[i,1] - points[j,1]))
            D[i, j] = d
            D[j, i] = d
    return D

# ---------- 2) 스윕 클러스터링 ----------
def sweep_cluster(points, demands, capacity):
    n = points.shape[0] - 1
    items = []
    for i in range(1, n + 1):
        dx = points[i,0] - points[0,0]
        dy = points[i,1] - points[0,1]
        ang = math.atan2(dy, dx)
        if ang < 0:
            ang += 2*math.pi
        items.append((i, ang, int(demands[i])))
    items.sort(key=lambda t: t[1])

    routes = []
    cur, load = [], 0
    for idx, _, dem in items:
        if load + dem <= capacity:
            cur.append(idx); load += dem
        else:
            routes.append(cur)
            cur = [idx]; load = dem
    if cur:
        routes.append(cur)
    return routes

# ---------- 3) 최근접 이웃 ----------
def nn_order(route: List[int], D: np.ndarray) -> List[int]:
    if not route: return []
    unvis = set(route)
    start = min(unvis, key=lambda k: D[0, k])
    order = [start]
    unvis.remove(start)
    while unvis:
        last = order[-1]
        nxt = min(unvis, key=lambda k: D[last, k])
        order.append(nxt)
        unvis.remove(nxt)
    return order

# ---------- (선택) 2-opt: 일단 패스 ----------
def two_opt(order: List[int], D: np.ndarray, max_iter: int = 1000) -> List[int]:
    return order

# ---------- 거리 ----------
def route_distance(order: List[int], D: np.ndarray) -> float:
    if not order: return 0.0
    dist = D[0, order[0]] + D[order[-1], 0]
    for i in range(len(order)-1):
        dist += D[order[i], order[i+1]]
    return float(dist)

# ---------- 저장 ----------
def save_solution(routes: List[List[int]], ids: List[str], demands: np.ndarray, out_csv: str):
    rows = []
    for ridx, r in enumerate(routes, start=1):
        for k, idx in enumerate(r, start=1):
            rows.append({
                "route_id": ridx,
                "visit_order": k,
                "point_id": ids[idx],
                "demand": int(demands[idx])
            })
    pd.DataFrame(rows).to_csv(out_csv, index=False)


ids, points, demands = load_data(DATA_CSV)
D = build_dist_matrix(points)
t0 = time.time()
routes = sweep_cluster(points, demands, CAPACITY)
routes = [nn_order(r, D) for r in routes]
total = sum(route_distance(r, D) for r in routes)
t1 = time.time()

print(f"노선 수={len(routes)}, 총 이동거리≈{total:.2f} km, 시간={t1 - t0:.4f}초")
save_solution(routes, ids, demands, OUT_CSV)
print(f"Saved: {OUT_CSV}")
