# visualize_submission_path.py
# 필요: matplotlib (pandas/np 없이 csv 표준 라이브러리만 사용)

import csv, math
import matplotlib.pyplot as plt

DATA_CSV = "data.csv"
SUBMISSION_CSV = "submission.csv"

def load_coords(data_csv):
    coords = {}
    towns_x, towns_y = [], []
    depot_id = "DEPOT"
    with open(data_csv, newline="", encoding="utf-8") as f:
        rdr = csv.DictReader(f)
        for r in rdr:
            pid = r["point_id"]
            x = float(r["x"]); y = float(r["y"])
            coords[pid] = (x, y)
            if pid != depot_id:
                towns_x.append(x); towns_y.append(y)
            else:
                depot_xy = (x, y)
    return coords, depot_id, depot_xy, towns_x, towns_y

def load_path(sub_csv):
    seq = []
    with open(sub_csv, newline="", encoding="utf-8") as f:
        rdr = csv.DictReader(f)
        for r in rdr:
            seq.append(r["point_id"])
    return seq

def dist(a, b, coords):
    xa, ya = coords[a]; xb, yb = coords[b]
    return math.hypot(xa - xb, ya - yb)

def split_routes_by_depot(seq, depot):
    """ DEPOT을 경계로 [DEPOT..DEPOT], [DEPOT..DEPOT], ... 세그먼트로 분할 """
    segments = []
    cur = []
    for pid in seq:
        cur.append(pid)
        if pid == depot and len(cur) > 1:
            segments.append(cur)
            cur = [depot]  # 다음 라우트 시작용(DEPOT으로 시작)
    if len(cur) > 1:
        segments.append(cur)
    return segments

if __name__ == "__main__":
    coords, DEPOT, depot_xy, towns_x, towns_y = load_coords(DATA_CSV)
    seq = load_path(SUBMISSION_CSV)

    # 총 거리
    total = 0.0
    for a, b in zip(seq[:-1], seq[1:]):
        total += dist(a, b, coords)

    # 라우트별 세그먼트
    segments = split_routes_by_depot(seq, DEPOT)
    route_dists = []
    for seg in segments:
        d = 0.0
        for a, b in zip(seg[:-1], seg[1:]):
            d += dist(a, b, coords)
        route_dists.append(d)

    # ---- 시각화 ----
    plt.figure(figsize=(8, 8))
    # 전체 마을 산점도
    plt.scatter(towns_x, towns_y, s=20, label="TOWNS", zorder=2)
    # DEPOT
    plt.scatter([depot_xy[0]], [depot_xy[1]], s=120, marker="s", label="DEPOT", zorder=3)

    # 라우트 그리기 (DEPOT~DEPOT마다 색 변경)
    cmap = plt.cm.get_cmap("tab20", max(20, len(segments)))
    for i, seg in enumerate(segments):
        xs = [coords[p][0] for p in seg]
        ys = [coords[p][1] for p in seg]
        plt.plot(xs, ys, linewidth=1.8, color=cmap(i % 20), label=f"Route {i+1}")

    plt.title(f"Submission Path (Routes={len(segments)}, Total≈{total:.2f})")
    plt.xlabel("x (km)")
    plt.ylabel("y (km)")
    plt.axis("equal")
    # 범례가 너무 길면 ncol 조절
    plt.legend(loc="upper right", fontsize=8, ncol=2)
    plt.tight_layout()
    plt.savefig("submission_path.png", dpi=150)
    plt.show()

    # 요약 출력
    print("=== Route Summary (DEPOT~DEPOT) ===")
    for i, d in enumerate(route_dists, 1):
        print(f"Route {i:2d}: distance={d:.3f}")
    print(f"TOTAL distance: {total:.3f}")
    print("Saved figure: submission_path.png")
