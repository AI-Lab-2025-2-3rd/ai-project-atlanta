
import math
import numpy as np
import pandas as pd
from pulp import LpProblem, LpMinimize, LpVariable, LpBinary, LpStatus, lpSum, PULP_CBC_CMD

CAPACITY = 25
DATA_CSV = "data.csv"
OUT_CSV = "solutionMILP.csv"
TIME_LIMIT_SEC = 600  

# ---- 0) 데이터 로드 (DEPOT은 point_id == 'DEPOT' 가정) ----
df = pd.read_csv(DATA_CSV)
depot = df[df["point_id"] == "DEPOT"].iloc[0]
depot_id = depot["point_id"]
cust = df[df["point_id"] != depot_id].reset_index(drop=True)

ids = [depot_id] + cust["point_id"].tolist()             # 0..N 라벨
coords = np.zeros((len(ids), 2), dtype=float)
coords[0] = [float(depot["x"]), float(depot["y"])]
coords[1:, 0] = cust["x"].astype(float).values
coords[1:, 1] = cust["y"].astype(float).values

demand = np.zeros(len(ids), dtype=int)
demand[1:] = cust["demand"].astype(int).values
N = len(ids) - 1
total_demand = int(demand.sum())

# ---- 1) 거리 행렬 ----
def dist(i, j):
    return float(math.hypot(coords[i,0]-coords[j,0], coords[i,1]-coords[j,1]))

D = {(i,j): (0.0 if i==j else dist(i,j)) for i in range(N+1) for j in range(N+1)}

# ---- 2) MILP 변수 ----
# x_ij ∈ {0,1} : i→j로 이동하면 1 (i≠j만 정의)
# f_ij ≥ 0     : i→j로 흐르는 "화물량"(single commodity flow)
prob = LpProblem("CVRP_SCF_Exact", LpMinimize)

x = {}
f = {}
for i in range(N+1):
    for j in range(N+1):
        if i == j:
            continue
        x[(i,j)] = LpVariable(f"x_{i}_{j}", lowBound=0, upBound=1, cat=LpBinary)
        f[(i,j)] = LpVariable(f"f_{i}_{j}", lowBound=0)  # 연속

# ---- 3) 목적함수: 총 거리 ----
prob += lpSum(D[(i,j)] * x[(i,j)] for i in range(N+1) for j in range(N+1) if i!=j)

# ---- 4) 방문 제약 (각 고객 정확히 한 번 in/out) ----
for j in range(1, N+1):
    prob += lpSum(x[(i,j)] for i in range(N+1) if i!=j) == 1   # 들어오는 간선 1개
    prob += lpSum(x[(j,k)] for k in range(N+1) if k!=j) == 1   # 나가는 간선 1개

# depot의 in/out 균형 (경로 수 동일)
prob += lpSum(x[(0,j)] for j in range(1, N+1)) == lpSum(x[(j,0)] for j in range(1, N+1))

# ---- 5) 용량/흐름 제약 (SCF) ----
# (a) 각 고객의 흐름 보존: 들어오는 흐름 - 나가는 흐름 = demand_j
for j in range(1, N+1):
    prob += lpSum(f[(i,j)] for i in range(N+1) if i!=j) - lpSum(f[(j,k)] for k in range(N+1) if k!=j) == demand[j]

# (b) depot의 흐름: 나가는 흐름 - 들어오는 흐름 = total_demand
prob += lpSum(f[(0,j)] for j in range(1, N+1)) - lpSum(f[(j,0)] for j in range(1, N+1)) == total_demand

# (c) 용량 연계: f_ij ≤ CAPACITY * x_ij  (차가 없는 간선엔 흐름 불가, 있으면 최대 용량)
for i in range(N+1):
    for j in range(N+1):
        if i == j: 
            continue
        prob += f[(i,j)] <= CAPACITY * x[(i,j)]

# ---- 6) 풀기 ----
solver = PULP_CBC_CMD(msg=True, timeLimit=TIME_LIMIT_SEC) if TIME_LIMIT_SEC > 0 else PULP_CBC_CMD(msg=True)
status = prob.solve(solver)
print("Status:", LpStatus[status])
print("Objective (total distance):", prob.objective.value())

# ---- 7) 해 추출: x_ij = 1인 간선들로 경로 복원 ----
# 부동소수 오차 방지: 0.5 기준으로 이진판정
succ = {i: [] for i in range(N+1)}
for i in range(N+1):
    for j in range(N+1):
        if i==j: continue
        if x[(i,j)].value() is not None and x[(i,j)].value() > 0.5:
            succ[i].append(j)

routes = []
used = set()

# depot에서 시작하는 모든 경로 추적
for j in succ[0]:
    if (0,j) in used:
        continue
    route = []
    cur = j
    used.add((0,j))
    while cur != 0:
        route.append(cur)
        nxts = succ[cur]
        if not nxts:
            break
        nxt = nxts[0]
        used.add((cur,nxt))
        cur = nxt
    routes.append(route)

# ---- 8) CSV 저장 (route_id, visit_order, point_id, demand) ----
rows = []
for ridx, r in enumerate(routes, start=1):
    for k, idx in enumerate(r, start=1):
        rows.append({
            "route_id": ridx,
            "visit_order": k,
            "point_id": ids[idx],
            "demand": int(demand[idx]),
        })
pd.DataFrame(rows).to_csv(OUT_CSV, index=False)
print(f"Saved: {OUT_CSV}")
