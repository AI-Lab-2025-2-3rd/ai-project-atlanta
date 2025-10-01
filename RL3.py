import csv, math, random, time
from collections import deque
from typing import List, Tuple

import numpy as np
import tensorflow as tf
from tensorflow.keras import layers, optimizers, Model

# ==== 상단 import 근처에 추가 ====
import os
try:
    import matplotlib.pyplot as plt
    _HAVE_PLT = True
except Exception:
    _HAVE_PLT = False


# ==== 보조 함수 두 개 추가 ====
def _save_metrics_csv(out_csv, rows):
    os.makedirs(os.path.dirname(out_csv), exist_ok=True)
    import csv as _csv
    header = ["episode", "total_dist", "best_dist", "steps", "routes", "avg_loss", "epsilon"]
    write_header = not os.path.exists(out_csv)
    with open(out_csv, "a", newline="", encoding="utf-8") as f:
        w = _csv.writer(f)
        if write_header:
            w.writerow(header)
        for r in rows:
            w.writerow([r[h] for h in header])

def _save_metrics_plot(out_png, hist_ep, hist_dist, hist_best):
    if not _HAVE_PLT:
        return
    os.makedirs(os.path.dirname(out_png), exist_ok=True)
    plt.figure(figsize=(7.5, 4.5))
    plt.plot(hist_ep, hist_dist, label="episode total distance")
    plt.plot(hist_ep, hist_best, label="best distance", linestyle="--")
    plt.xlabel("episode")
    plt.ylabel("distance (lower is better)")
    plt.title("CVRP RL training progress")
    plt.legend()
    plt.tight_layout()
    plt.savefig(out_png, dpi=150)
    plt.close()

def _save_best_path_csv(out_csv, path_indices):
    os.makedirs(os.path.dirname(out_csv), exist_ok=True)
    import csv as _csv
    with open(out_csv, "w", newline="", encoding="utf-8") as f:
        w = _csv.writer(f)
        w.writerow(["node_idx"])  # 0=DEPOT 포함 전체 방문 순서
        for idx in path_indices:
            w.writerow([idx])

# ==== 루트 분리 유틸 ====
def _split_routes_from_path(path_indices):
    # path 예: [0, 22, 16, 24, 21, 15, 0, 68, 73, ... , 0]
    routes = []
    cur = []
    for idx in path_indices:
        if idx == 0:
            if cur:
                routes.append(cur)
                cur = []
            # DEPOT은 루트 경계 표시만: 내부에 넣지 않음
        else:
            cur.append(idx)
    if cur:  # 마지막에 DEPOT으로 끝나지 않은 경우 보정
        routes.append(cur)
    return routes

# ==== 경로 시각화 (루트별 색상) ====
def _save_best_path_plot(out_png, points, path_indices):
    if not _HAVE_PLT:
        return
    os.makedirs(os.path.dirname(out_png), exist_ok=True)

    xs, ys = points[:, 0], points[:, 1]

    routes = _split_routes_from_path(path_indices)
    # 총거리 계산(각 루트: 0 -> r... -> 0)
    total = 0.0
    for r in routes:
        seq = [0] + r + [0]
        for i in range(len(seq) - 1):
            a, b = seq[i], seq[i+1]
            total += float(np.hypot(xs[b] - xs[a], ys[b] - ys[a]))

    import matplotlib.pyplot as plt
    plt.figure(figsize=(7.2, 7.2))

    # 노드
    plt.scatter(xs[1:], ys[1:], s=18, alpha=0.9, label="TOWNS")
    plt.scatter([xs[0]], [ys[0]], s=80, marker="s", label="DEPOT")

    # 컬러맵에서 루트별 색
    cmap = plt.get_cmap("tab20")
    for ridx, r in enumerate(routes, start=1):
        color = cmap((ridx - 1) % 20)
        seq = [0] + r + [0]
        px = [xs[i] for i in seq]
        py = [ys[i] for i in seq]
        # 라인 + 점
        plt.plot(px, py, linewidth=1.6, color=color, label=f"Route {ridx}")
        plt.scatter([xs[i] for i in r], [ys[i] for i in r], s=20, color=color)

    plt.title(f"Submission Path (Routes={len(routes)}, Total≈{total:.2f})")
    plt.xlabel("x (km)"); plt.ylabel("y (km)")
    plt.axis("equal")
    plt.legend(ncol=2, fontsize=8)
    plt.tight_layout()
    plt.savefig(out_png, dpi=160)
    plt.close()



# ==== data.csv 로드 (단일 파일 / 정수형 / 헤더 1줄) ====
# towns: [(idx, x, y, demand)]  형태. idx=0 이 반드시 DEPOT.
towns = []

with open("data.csv", "r", newline="", encoding="utf-8") as f:
    rdr = csv.reader(f)
    header = next(rdr)  # ['point_id','x','y','demand'] 헤더 1줄 건너뛰기
    rows = [row for row in rdr if row]

# DEPOT 먼저 넣기 (idx=0)
for row in rows:
    if row[0] == "DEPOT":
        towns.append((0, int(row[1]), int(row[2]), int(row[3])))
        break

# 나머지 마을들 1..N 부여
idx = 1
for row in rows:
    if row[0] == "DEPOT":
        continue
    towns.append((idx, int(row[1]), int(row[2]), int(row[3])))
    idx += 1

# 마을을 돌면서 하나씩 제거함. 이때 1회 이동시 요구량 최대 75까지 가능
# 더 이상 나눠줄 수 없으면 처음 위치로 돌아옴. depot에서 선물 챙겨서 새로운 route 만들기.
# 제약조건 있는 조합 최적화 문제로 간주, RL 이용해 풀어보자

SEED = 0
EPISODES = 400         
MEM_CAP = 20000
BATCH = 64
GAMMA = 0.99
LR = 1e-3
EPS_START, EPS_END = 1.0, 0.05
EPS_DECAY_STEPS = 5000  
TARGET_SYNC = 500     
COORD_SCALE = 100.0   

TEMP_START, TEMP_END = 1.5, 0.05     # 시작/끝 온도
TEMP_DECAY_STEPS = 20000              # τ가 TEMP_END에 도달하는 스텝 수


random.seed(SEED)
np.random.seed(SEED)
tf.random.set_seed(SEED)

def euclid(a, b):
    return float(math.hypot(a[1] - b[1], a[2] - b[2]))


#마스크를 이용해 불법 행동을 제거하는 그런 DQN 모델을 쓸거임(조합최적화 문제 풀이에서 일반적으로 사용된다고 함)

# ===== Masked DQN for CVRP: 학습용 =====
CAPACITY = 25  # 썰매 용량

# towns: [(idx, x, y, demand), ...] 이라고 가정한다 (0번째는 DEPOT).
# euclid(a,b)는 (idx,x,y,d) 튜플용이므로 아래 env에서는 좌표 배열로 따로 씀.

# ---------- 특징/마스크 ----------
COORD_SCALE = 100.0  # 좌표/거리 정규화

def build_features_and_mask(points, demands, visited, cur_idx, remcap):
    """
    points: (Np,2) float32, demands: (Np,) int
    visited: (Np,) bool, cur_idx: int, remcap: int
    반환:
      feats: (Np, 7) float32  [dx,dy,dist, demand/C, remcap/C, visited, is_depot]
      mask : (Np,)  bool      True=선택 가능
    규칙:
      - 이미 방문했거나 demand>remcap인 고객 금지
      - 가능한 고객이 1개 이상이면 DEPOT 금지
      - 모든 고객 방문 완료면 DEPOT만 허용
      - cur==DEPOT이고 feasible 고객 있으면 DEPOT 금지
    """
    Np = points.shape[0]
    dx = points[:, 0] - points[cur_idx, 0]
    dy = points[:, 1] - points[cur_idx, 1]
    dist = np.hypot(dx, dy) / COORD_SCALE

    demand_norm = demands.astype(np.float32) / CAPACITY
    remcap_norm = np.full((Np,), remcap / CAPACITY, dtype=np.float32)
    visited_flag = visited.astype(np.float32)
    is_depot = np.zeros((Np,), dtype=np.float32); is_depot[0] = 1.0

    feats = np.stack([
        dx / COORD_SCALE, dy / COORD_SCALE, dist,
        demand_norm, remcap_norm, visited_flag, is_depot
    ], axis=1).astype(np.float32)

    mask = np.ones((Np,), dtype=bool)
    # 고객 마스킹
    for j in range(1, Np):
        if visited[j] or demands[j] > remcap:
            mask[j] = False
    feasible = np.any(mask[1:])
    # DEPOT 마스킹
    mask[0] = not feasible
    # 모든 고객 방문완료 → DEPOT만 허용
    if np.all(visited[1:]):
        mask[:] = False
        mask[0] = True
    # 시작 직후/현재 DEPOT인데 feasible 있으면 DEPOT 금지
    if cur_idx == 0 and feasible:
        mask[0] = False

    if not mask.any():  # 안전핀
        mask[0] = True
    return feats, mask

# ---------- 환경 ----------
class CVRPEnv:
    def __init__(self, towns):
        # towns -> ids, points, demands
        self.ids = [int(t[0]) for t in towns]
        self.points = np.array([(float(t[1]), float(t[2])) for t in towns], dtype=np.float32)
        self.demands = np.array([int(t[3]) for t in towns], dtype=np.int32)
        self.Np = len(self.ids)  # 0..N (0=DEPOT)
        assert self.ids[0] == 0, "0번째가 DEPOT이어야 합니다(인덱스 0)."
        self.reset()

    def reset(self):
        self.visited = np.zeros((self.Np,), dtype=bool)
        self.cur = 0
        self.remcap = CAPACITY
        self.path = [0]  # 인덱스 경로(DEPOT부터)
        feats, mask = build_features_and_mask(self.points, self.demands, self.visited, self.cur, self.remcap)
        return feats, mask

    def step(self, a_idx: int):
        # 보상: -거리
        d = float(np.hypot(
            self.points[self.cur,0] - self.points[a_idx,0],
            self.points[self.cur,1] - self.points[a_idx,1]
        ))
        reward = - (d / 1.0)  # 스케일 조정 가능

        # 전이
        if a_idx == 0:
            self.cur = 0
            self.remcap = CAPACITY
        else:
            self.cur = a_idx
            self.remcap -= int(self.demands[a_idx])
            self.visited[a_idx] = True

        self.path.append(a_idx)

        feats, mask = build_features_and_mask(self.points, self.demands, self.visited, self.cur, self.remcap)
        done = bool(np.all(self.visited[1:]) and self.cur == 0)
        return (feats, mask), reward, done

# ---------- Q 네트워크(노드별 공유 MLP) ----------
class QNet(Model):
    def __init__(self, hidden=128):
        super().__init__()
        self.d1 = layers.Dense(hidden, activation="relu")
        self.d2 = layers.Dense(hidden, activation="relu")
        self.o  = layers.Dense(1, activation=None)  # 각 노드의 Q

    def call(self, feats):   # feats: (B, Np, 7)
        x = self.d1(feats)
        x = self.d2(x)
        q = self.o(x)                       # (B,Np,1)
        return tf.squeeze(q, axis=-1)       # (B,Np)

# ---------- 에이전트 (Masked DQN) ----------
class DQNAgent:
    def __init__(self, Np, hidden=128, lr=LR):
        self.Np = Np
        self.q = QNet(hidden)
        self.t = QNet(hidden)
        # 빌드
        dummy = tf.zeros((1, Np, 7), dtype=tf.float32)
        self.q(dummy); self.t(dummy)
        self.t.set_weights(self.q.get_weights())
        self.opt = optimizers.Adam(lr)
        self.mem = deque(maxlen=MEM_CAP)
        self.eps = EPS_START
        self.total_steps = 0
        self.tau = TEMP_START

    def select_action(self, feats, mask, greedy=False):
        # feats: (Np,7), mask: (Np,)
        q = self.q(tf.convert_to_tensor(feats[None, ...]))[0].numpy()  # (Np,)
        q_masked = q.copy(); q_masked[~mask] = -1e9

        if greedy:
            a = int(np.argmax(q_masked))
            return a, float(q[a])

        # ---- Boltzmann(softmax) sampling with temperature τ ----
        idxs = np.where(mask)[0]
        logits = q_masked[idxs] / max(1e-8, self.tau)
        logits = logits - np.max(logits)              # overflow 방지
        probs = np.exp(logits); probs /= np.sum(probs)
        a = int(np.random.choice(idxs, p=probs))
        return a, float(q[a])


    def push(self, s_feats, s_mask, a, r, ns_feats, ns_mask, done):
        self.mem.append((
            s_feats.astype(np.float32),
            s_mask.astype(np.bool_),
            int(a), float(r),
            ns_feats.astype(np.float32),
            ns_mask.astype(np.bool_),
            bool(done)
        ))

    def train_step(self, batch_size=BATCH):
        if len(self.mem) < batch_size:
            return 0.0
        batch = random.sample(self.mem, batch_size)
        s_f = np.stack([b[0] for b in batch])       # (B,Np,7)
        s_m = np.stack([b[1] for b in batch])       # (B,Np)
        a   = np.array([b[2] for b in batch], dtype=np.int32)  # (B,)
        r   = np.array([b[3] for b in batch], dtype=np.float32)
        ns_f= np.stack([b[4] for b in batch])
        ns_m= np.stack([b[5] for b in batch])
        d   = np.array([b[6] for b in batch], dtype=np.float32)

        with tf.GradientTape() as tape:
            q_all = self.q(s_f)                           # (B,Np)
            q_sa = tf.gather(q_all, a[:,None], batch_dims=1)[:,0]  # (B,)

            q_next = self.t(ns_f).numpy()                 # (B,Np)
            q_next[~ns_m] = -1e9                          # 마스크 적용
            max_next = np.max(q_next, axis=1).astype(np.float32)
            target = r + (1.0 - d) * GAMMA * max_next     # (B,)

            loss = tf.reduce_mean(tf.square(q_sa - target))

        grads = tape.gradient(loss, self.q.trainable_variables)
        self.opt.apply_gradients(zip(grads, self.q.trainable_variables))
        return float(loss.numpy())

    def sync_target(self):
        self.t.set_weights(self.q.get_weights())

    def decay_temperature(self):
        self.total_steps += 1
        frac = min(1.0, self.total_steps / TEMP_DECAY_STEPS)
        self.tau = TEMP_START + (TEMP_END - TEMP_START) * frac


# ---------- 학습 루프 ----------
def train(towns, episodes=EPISODES, log_every=1):
    env = CVRPEnv(towns)
    agent = DQNAgent(Np=env.Np, hidden=128, lr=LR)

    best_dist = float("inf")
    best_path = None

    # 진행 기록(그래프용)
    hist_ep, hist_dist, hist_best = [], [], []

    # artifacts 폴더
    ART_DIR = "artifacts"
    METRIC_CSV = os.path.join(ART_DIR, "metrics.csv")
    METRIC_PNG = os.path.join(ART_DIR, "metrics.png")

    for ep in range(1, episodes + 1):
        feats, mask = env.reset()
        done = False
        ep_reward = 0.0
        steps = 0
        ep_loss_sum = 0.0
        ep_updates = 0

        while not done and steps < 4000:
            a, _ = agent.select_action(feats, mask, greedy=False)
            (nf, nm), r, done = env.step(a)

            agent.push(feats, mask, a, r, nf, nm, done)
            loss = agent.train_step(BATCH)
            if loss:
                ep_loss_sum += float(loss)
                ep_updates += 1

            feats, mask = nf, nm
            ep_reward += r
            steps += 1

            agent.decay_temperature()
            if agent.total_steps % TARGET_SYNC == 0:
                agent.sync_target()

        # 에피소드 결과
        total_dist = -ep_reward                     # 총 이동거리(작을수록 좋음)
        avg_loss = (ep_loss_sum / ep_updates) if ep_updates > 0 else 0.0
        route_cnt = max(0, env.path.count(0) - 1)  # DEPOT 복귀 횟수

        # 기록
        hist_ep.append(ep)
        hist_dist.append(total_dist)
        if total_dist < best_dist:
            best_dist = total_dist
            best_path = env.path[:]
        hist_best.append(best_dist)

        # 콘솔 로그
        if ep % log_every == 0:
            print(
                f"[EP {ep:4d}] "
                f"steps={steps:4d} "
                f"tau={agent.tau:.3f} "
                f"reward_sum={ep_reward:.2f} "
                f"dist≈{total_dist:.2f} "
                f"routes={route_cnt} "
                f"avg_loss={avg_loss:.5f} "
                f"best≈{best_dist:.2f}"
            )

        # === 10 에피소드마다: 메트릭/그래프/베스트 경로 저장 ===
        if ep % 10 == 0:
            # 메트릭 CSV에 누적 저장
            rows = [{
                "episode": ep_i,
                "total_dist": d_i,
                "best_dist": b_i,
                "steps": None,          # 간단히 None; 원하면 per-ep steps 저장 로직 추가 가능
                "routes": None,
                "avg_loss": None,
                "epsilon": round(agent.tau, 6),
            } for ep_i, d_i, b_i in zip(hist_ep, hist_dist, hist_best)]
            _save_metrics_csv(METRIC_CSV, rows)

            # 그래프 PNG 저장(덮어쓰기)
            _save_metrics_plot(METRIC_PNG, hist_ep, hist_dist, hist_best)

            # 현재 베스트 경로 저장
            if best_path is not None:
                path_csv = os.path.join(ART_DIR, f"best_path_ep{ep}.csv")
                path_png = os.path.join(ART_DIR, f"best_path_ep{ep}.png")
                _save_best_path_csv(path_csv, best_path)
                _save_best_path_plot(path_png, env.points, best_path)

    print(f"\nBest distance (approx): {best_dist:.2f}")
    # 마지막으로 한 번 더 갱신 저장(종료 시점)
    rows = [{
        "episode": ep_i,
        "total_dist": d_i,
        "best_dist": b_i,
        "steps": None,
        "routes": None,
        "avg_loss": None,
        "epsilon": None,
    } for ep_i, d_i, b_i in zip(hist_ep, hist_dist, hist_best)]
    _save_metrics_csv(os.path.join(ART_DIR, "metrics.csv"), rows)
    _save_metrics_plot(os.path.join(ART_DIR, "metrics.png"), hist_ep, hist_dist, hist_best)
    if best_path is not None:
        _save_best_path_csv(os.path.join(ART_DIR, f"best_path_final.csv"), best_path)
        _save_best_path_plot(os.path.join(ART_DIR, f"best_path_final.png"), env.points, best_path)

    return best_dist, best_path


# ---------- (옵션) 그리디 평가 ----------
def evaluate_greedy(towns, agent: DQNAgent):
    env = CVRPEnv(towns)
    feats, mask = env.reset()
    done = False; total = 0.0
    while not done and len(env.path) < 4000:
        a, _ = agent.select_action(feats, mask, greedy=True)
        (nf, nm), r, done = env.step(a)
        total += -r
        feats, mask = nf, nm
    return total, env.path

# ===== 실행(학습) =====
if __name__ == "__main__":
    best_dist, best_path = train(towns, episodes=EPISODES)
    print("Best distance (approx):", best_dist)
    # best_path 는 인덱스 시퀀스(0=DEPOT 포함). 제출 저장은 별도 파일에서 처리하면 됨.
