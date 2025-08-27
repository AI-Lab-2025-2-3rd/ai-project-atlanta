import csv, math, random, time
from typing import List, Dict, Tuple

CAPACITY = 25
DATA_CSV = "data.csv"
OUT_CSV = "solution.csv"
TIME_LIMIT_SEC = 5
SEED = 0

# ------------------ 데이터 로드 ------------------
def load_data(path: str):
    with open(path, newline="", encoding="utf-8") as f:
        rdr = csv.DictReader(f)
        rows = list(rdr)

    coords: Dict[str, Tuple[float, float]] = {}
    demand: Dict[str, int] = {}
    for r in rows:
        pid = r["point_id"]
        coords[pid] = (float(r["x"]), float(r["y"]))
        demand[pid] = int(r["demand"])

    depot_id = "DEPOT"
    towns = [r["point_id"] for r in rows if r["point_id"] != depot_id]
    return depot_id, coords, demand, towns

# ------------------ 거리 함수(캐시) ------------------
def make_distance_fn(coords: Dict[str, Tuple[float,float]]):
    cache: Dict[Tuple[str,str], float] = {}
    def dist(a: str, b: str) -> float:
        if a == b: return 0.0
        key = (a, b)
        if key in cache: return cache[key]
        xa, ya = coords[a]; xb, yb = coords[b]
        d = math.hypot(xa - xb, ya - yb)
        cache[key] = d
        cache[(b, a)] = d
        return d
    return dist

# ------------------ 초기 해(랜덤) ------------------
def make_initial_routes(towns: List[str], demand: Dict[str,int], capacity: int) -> List[List[str]]:
    towns = towns[:]                 # 사본
    random.shuffle(towns)            # 랜덤 순열
    routes: List[List[str]] = []
    cur, load = [], 0
    for pid in towns:
        d = demand[pid]
        if load + d <= capacity:
            cur.append(pid); load += d
        else:
            routes.append(cur)
            cur, load = [pid], d
    if cur: routes.append(cur)
    return routes

# ------------------ 유틸 ------------------
def route_load(route: List[str], demand: Dict[str,int]) -> int:
    return sum(demand[p] for p in route)

def total_distance(routes: List[List[str]], depot: str, dist) -> float:
    tot = 0.0
    for r in routes:
        if not r: continue
        tot += dist(depot, r[0])
        for a, b in zip(r, r[1:]):
            tot += dist(a, b)
        tot += dist(r[-1], depot)
    return tot

def deep_copy_routes(routes: List[List[str]]) -> List[List[str]]:
    return [r[:] for r in routes]

def cleanup_empty_routes(routes: List[List[str]]):
    i = 0
    while i < len(routes):
        if not routes[i]:
            routes.pop(i)
        else:
            i += 1

# ------------------ 무브 1: 경로 내 순서 변경(스왑) ------------------
def move_swap_in_route(routes: List[List[str]]):
    if not routes: return False
    ridx = random.randrange(len(routes))
    r = routes[ridx]
    if len(r) < 2: return False
    i, j = random.sample(range(len(r)), 2)
    r[i], r[j] = r[j], r[i]
    return True

# ------------------ 무브 2: 경로 간 리로케이트(한 고객 이동) ------------------
def move_relocate_between_routes(routes: List[List[str]], demand: Dict[str,int], capacity: int):
    if len(routes) < 2: return False
    src = random.randrange(len(routes))
    if not routes[src]: return False
    dst = random.randrange(len(routes))
    if src == dst: return False

    i = random.randrange(len(routes[src]))
    pid = routes[src].pop(i)

    j = random.randrange(len(routes[dst]) + 1)
    routes[dst].insert(j, pid)

    # 용량 체크(안 되면 롤백)
    if route_load(routes[dst], demand) > capacity:
        routes[dst].pop(j)
        routes[src].insert(i, pid)
        return False
    return True

# ------------------ 무브 3: 경로 간 스왑(서로 다른 경로의 고객 교환) ------------------
def move_swap_between_routes(routes: List[List[str]], demand: Dict[str,int], capacity: int):
    if len(routes) < 2: return False
    a = random.randrange(len(routes))
    b = random.randrange(len(routes))
    if a == b or not routes[a] or not routes[b]: return False

    ia = random.randrange(len(routes[a]))
    ib = random.randrange(len(routes[b]))

    pa = routes[a][ia]
    pb = routes[b][ib]

    # 스왑 적용
    routes[a][ia], routes[b][ib] = pb, pa

    # 용량 체크(안 되면 롤백)
    if route_load(routes[a], demand) > capacity or route_load(routes[b], demand) > capacity:
        routes[a][ia], routes[b][ib] = pa, pb
        return False
    return True

# ------------------ 랜덤 탐색(힐클라임) ------------------
def random_search(depot: str, coords, demand, towns, time_limit_sec=5, capacity=25):
    dist = make_distance_fn(coords)

    best_routes = make_initial_routes(towns, demand, capacity)
    best_cost = total_distance(best_routes, depot, dist)

    routes = deep_copy_routes(best_routes)
    cost = best_cost

    start = time.time()
    moves = [
        lambda r: move_swap_in_route(r),
        lambda r: move_relocate_between_routes(r, demand, capacity),
        lambda r: move_swap_between_routes(r, demand, capacity),
    ]

    while time.time() - start < time_limit_sec:
        backup = deep_copy_routes(routes)
        mv = random.choice(moves)
        ok = mv(routes)
        if not ok:
            routes = backup
            continue

        cleanup_empty_routes(routes)
        new_cost = total_distance(routes, depot, dist)

        # 개선되면 채택, 아니면 롤백
        if new_cost + 1e-9 < cost:
            cost = new_cost
            if cost + 1e-9 < best_cost:
                best_cost = cost
                best_routes = deep_copy_routes(routes)
        else:
            routes = backup

    return best_routes, best_cost

# ------------------ 저장 ------------------
def save_submission_point_path(routes, depot, out_path="submission.csv"):
    seq = [depot]
    for r in routes:
        if not r:
            continue
        seq.extend(r)
        seq.append(depot)  # 각 라우트 끝마다 DEPOT

    with open(out_path, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["point_id"])
        for pid in seq:
            w.writerow([pid])


# ------------------ 실행 ------------------
if __name__ == "__main__":
    random.seed(SEED)
    depot, coords, demand, towns = load_data(DATA_CSV)

    best_routes, best_cost = random_search(
        depot=depot,
        coords=coords,
        demand=demand,
        towns=towns,
        time_limit_sec=TIME_LIMIT_SEC,
        capacity=CAPACITY,
    )

    print(f"routes={len(best_routes)}, total_distance≈{best_cost:.3f}")
    save_submission_point_path(best_routes, depot, out_path="submission.csv")
    print(f"Saved: submission.csv")
