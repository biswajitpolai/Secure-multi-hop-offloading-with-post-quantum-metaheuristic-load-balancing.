#!/usr/bin/env python3
"""
Secure trust-aware task offloading in Vehicular Edge Computing (VEC)
Proposed : SAC (discrete, trust-masked next-hop/offload policy) + DGSO scheduler
Baselines: DQN, DDQN, A3C (+ least-loaded server scheduling, no trust mechanism)

Everything is simulated from scratch (mobility, path loss, rates, energy, blackhole
attackers, trust/EWMA/conflict resolution, queues at edge servers) and the RL agents
are really trained in the same environment.

Usage
  pip install numpy torch matplotlib
  python vec_sim.py --smoke                 # ~1-2 min sanity run
  python vec_sim.py --quick                 # ~15-25 min, rough curves
  python vec_sim.py                         # full run (hours on a laptop CPU)
  python vec_sim.py --exp dgso              # only the DGSO optimisation study
"""
import argparse, copy, json, math, os, threading, time
from collections import defaultdict, deque
import numpy as np
import torch, torch.nn as nn, torch.nn.functional as F, torch.optim as optim
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

torch.set_num_threads(1)

# ----------------------------------------------------------------------------------
# Parameters (Table 1 / Table 2 of the paper)
# ----------------------------------------------------------------------------------
CFG = dict(
    area=2000.0, dt=0.5, steps=80, sched_every=2,            # 40 s simulated, 1 s DGSO window
    v_range=(10, 60), v2v_range=250.0, v2i_range=500.0,
    n_rsu=4, k_srv=4,
    bw=10e6, noise_dbm=-100.0, eta=2.7, fc_ghz=5.9,
    pt=(0.1, 2.0), pt_rsu=2.0, p_rx=0.1,
    local_f=(1e9, 5e9), edge_f=(10e9, 25e9),
    kappa=1e-27, kappa_edge=1e-28,                           # E = kappa * f^2 * C  (J)
    p_gen=0.05,                                              # task prob. per vehicle per step
    size_mu=math.log(8.0), size_sigma=1.2, size_lo=1.0, size_hi=1000.0,   # MB, clipped log-normal
    cpb=400.0,                                               # CPU cycles per byte
    dl_lo=1.5, dl_hi=6.0, dl_per_mb=0.03,
    backhaul=1e9, ascon_veh=1e8, ascon_srv=1e9,              # bit/s link, byte/s Ascon speed
    alpha=0.7, tmin=0.6, lmin=5.0, beta=0.6, conflict_thr=0.3,
    drop_p=0.9, loss_p=0.02, hmax=8,
    dgso_N=50, dgso_iters=200, dgso_iters_train=25, dgso_rp=0.3,
    dgso_w=(0.4, 0.3, 0.2, 0.1),
)
SD, NA, NC = 32, 6, 4          # state dim, actions (local, RSU, 4 candidates), #candidates
GAMMA = 0.95
METHODS = ["SAC+DGSO", "DQN", "DDQN", "A3C"]


def pl_v2v(d):   # log-distance, eta = 2.7, PL0 = free-space loss at 1 m @5.9GHz
    return 47.9 + 10 * CFG["eta"] * np.log10(np.maximum(d, 1.0))


def pl_v2i(d):   # WINNER II B1 (urban)
    return 22.7 * np.log10(np.maximum(d, 1.0)) + 41.0 + 20 * np.log10(CFG["fc_ghz"] / 5.0)


def link_rate(pt, pl_db):   # Shannon rate (bit/s)
    pr = 10 * np.log10(pt * 1000.0) - pl_db
    snr = 10 ** ((pr - CFG["noise_dbm"]) / 10)
    return CFG["bw"] * np.log2(1 + snr)


# ----------------------------------------------------------------------------------
# DGSO scheduler (vectorised over the population)
# ----------------------------------------------------------------------------------
def greedy_eft(c):
    """Earliest-finish-time greedy (used as DGSO warm-start seed / study baseline)."""
    M, K = c["ex"].shape
    avail, a = c["busy"].copy(), np.zeros(M, int)
    for m in range(M):
        fin = np.maximum(avail, c["rdy"][m]) + c["ex"][m]
        k = int(fin.argmin()); a[m] = k; avail[k] = fin[k]
    return a


def least_loaded(c):
    """Baseline scheduler used by DQN/DDQN/A3C: send task to currently least-loaded server."""
    M, K = c["ex"].shape
    avail, a = c["busy"].copy(), np.zeros(M, int)
    for m in range(M):
        k = int(avail.argmin()); a[m] = k
        avail[k] = max(avail[k], c["rdy"][m, k]) + c["ex"][m, k]
    return a


class DGSO:
    def __init__(self, N=50, iters=200, rp=0.3, w=(0.4, 0.3, 0.2, 0.1), warm=True, rng=None):
        self.N, self.iters, self.rp, self.w, self.warm = N, iters, rp, w, warm
        self.rng = rng or np.random.default_rng(0)
        self.prev, self.hist = {}, None

    # ---- vectorised queue simulation of N candidate assignments (tasks in EDF order)
    def _sim(self, a, c):
        K = c["ex"].shape[1]
        oh = np.eye(K)[a]                                    # (N,M,K)
        ex = oh * c["ex"][None]
        cum = np.cumsum(ex, 1)
        fin = (np.maximum(c["busy"][None, None, :] + cum, c["rdy"][None] + ex) * oh).sum(2)
        load = np.maximum(c["busy"] - c["t0"], 0)[None, :] + cum[:, -1, :]
        return oh, fin, load

    def _repair(self, a, c):
        oh, fin, load = self._sim(a, c)
        # capacity pass: move least-urgent tasks off overloaded servers to least loaded one
        over = load > 1.5 * load.mean(1, keepdims=True) + 1e-9
        on_over = np.take_along_axis(over, a, 1)
        movable = on_over & (c["dl"][None] >= np.median(c["dl"])) & (self.rng.random(a.shape) < 0.5)
        a2 = np.where(movable, load.argmin(1)[:, None], a)
        # deadline pass: tasks projected late -> server giving earliest finish
        oh, fin, load = self._sim(a2, c)
        late = fin > c["dl"][None]
        alt = np.maximum(c["t0"] + load[:, None, :], c["rdy"][None]) + c["ex"][None]
        return np.where(late, alt.argmin(2), a2)

    def _eval(self, Y, c):
        K = c["ex"].shape[1]
        a0 = Y.argmax(2)                                     # decode
        a = self._repair(a0, c)                              # repair
        ch = a != a0
        if ch.any():
            Y = np.where(ch[..., None], 0.1 + 0.8 * np.eye(K)[a], Y)
        oh, fin, load = self._sim(a, c)
        delay = (fin - c["ra"][None]).mean(1) / c["dref"]
        imb = load.std(1) / (load.mean(1) + 0.5)
        miss = (fin > c["dl"][None]).mean(1)
        en = (oh * c["eE"][None]).sum((1, 2)) / (c["eE"].mean(1).sum() + 1e-12)
        w = self.w
        Fv = w[0] * delay + w[1] * imb + w[2] * miss + w[3] * en
        return Y, a, Fv, load

    def optimize(self, c, key=0):
        M, K = c["ex"].shape
        N, rng = self.N, self.rng
        Y = rng.random((N, M, K))
        if self.warm:
            if key in self.prev:                             # rolling-horizon seed
                P = self.prev[key]; m = min(M, P.shape[0]); Y[0, :m] = P[:m]
            Y[1] = 0.1 + 0.8 * np.eye(K)[greedy_eft(c)]
        Y, a, Fv, load = self._eval(Y, c)
        hist = []
        for t in range(self.iters):
            A = 1.0 - t / self.iters                         # glide coefficient decay
            order = np.argsort(Fv)
            L = Y[order[0]].copy()                           # leader
            r1, r2, r3 = (rng.integers(0, N, N) for _ in range(3))
            phi = rng.random((N, 1, 1))
            Yn = Y + A * rng.random((N, 1, 1)) * (L[None] - Y) + 0.5 * A * phi * (Y[r1] - Y[r2])
            rank = np.empty(N, int); rank[order] = np.arange(N)
            weak = (rank >= N // 2) & (rng.random(N) < self.rp)       # weak-agent replacement
            if weak.any():
                Yn[weak] = Y[r1[weak]] + rng.random((weak.sum(), 1, 1)) * (Y[r2[weak]] - Y[r3[weak]])
            jump = np.where(rng.random(N) < A)[0]                      # discrete glide jump
            if len(jump):
                mm = rng.integers(0, M, len(jump)); kk = load.argmin(1)[jump]
                Yn[jump, mm, :] = 0.1; Yn[jump, mm, kk] = 0.9
            Yn[order[0]] = L
            Yn = np.clip(Yn, 0.0, 1.0)
            Yn, an, Fn, ln = self._eval(Yn, c)
            imp = Fn < Fv                                    # greedy acceptance
            Y[imp], Fv[imp], a[imp], load[imp] = Yn[imp], Fn[imp], an[imp], ln[imp]
            hist.append(Fv.min())
        b = int(Fv.argmin())
        self.prev[key] = Y[b].copy(); self.hist = np.array(hist)
        return a[b]


# ----------------------------------------------------------------------------------
# RL agents  (all use action masks)
# ----------------------------------------------------------------------------------
def mlp(i, o, h=128):
    return nn.Sequential(nn.Linear(i, h), nn.ReLU(), nn.Linear(h, h), nn.ReLU(), nn.Linear(h, o))


class Policy:
    training = False
    def act(self, x, m): raise NotImplementedError
    def learn(self, trajs): pass
    def end_episode(self): pass


class OffPolicy(Policy):
    def __init__(self):
        self.buf = deque(maxlen=50000); self.nup = 0; self.updates_per_call = 10; self.bs = 64

    def learn(self, trajs):
        for tr in trajs:
            for k, e in enumerate(tr):
                done = k == len(tr) - 1
                x2 = np.zeros(SD, np.float32) if done else tr[k + 1]["x"]
                m2 = np.zeros(NA, bool) if done else tr[k + 1]["m"]
                self.buf.append((e["x"], e["a"], e["r"], x2, m2, done))
        if len(self.buf) < 256: return
        for _ in range(self.updates_per_call):
            idx = np.random.randint(0, len(self.buf), self.bs)
            b = [self.buf[i] for i in idx]
            X = torch.from_numpy(np.stack([e[0] for e in b]))
            A = torch.tensor([e[1] for e in b]).unsqueeze(1)
            R = torch.tensor([e[2] for e in b], dtype=torch.float32).unsqueeze(1)
            X2 = torch.from_numpy(np.stack([e[3] for e in b]))
            M2 = torch.from_numpy(np.stack([e[4] for e in b]))
            D = torch.tensor([float(e[5]) for e in b]).unsqueeze(1)
            self._update(X, A, R, X2, M2, D); self.nup += 1


class DQNAgent(OffPolicy):
    def __init__(self, double=False):
        super().__init__()
        self.q = mlp(SD, NA); self.qt = copy.deepcopy(self.q)
        self.opt = optim.Adam(self.q.parameters(), 1e-3); self.double = double; self.eps = 1.0

    def act(self, x, m):
        if self.training and np.random.rand() < self.eps:
            return int(np.random.choice(np.where(m)[0]))
        with torch.no_grad():
            q = self.q(torch.from_numpy(x)).numpy()
        return int(np.where(m, q, -1e9).argmax())

    def _update(self, X, A, R, X2, M2, D):
        q = self.q(X).gather(1, A)
        with torch.no_grad():
            qt = self.qt(X2)
            if self.double:
                a2 = self.q(X2).masked_fill(~M2, -1e9).argmax(1, keepdim=True)
                v = qt.gather(1, a2)
            else:
                v = qt.masked_fill(~M2, -1e9).max(1, keepdim=True)[0]
            y = R + GAMMA * (1 - D) * v
        loss = F.smooth_l1_loss(q, y)
        self.opt.zero_grad(); loss.backward(); nn.utils.clip_grad_norm_(self.q.parameters(), 5.0); self.opt.step()
        if self.nup % 200 == 0: self.qt.load_state_dict(self.q.state_dict())

    def end_episode(self): self.eps = max(0.05, self.eps * 0.97)


class SACAgent(OffPolicy):
    """Discrete SAC with action masking (masks include the trust/link-lifetime safety filter)."""
    def __init__(self, alpha=0.05, tau=0.01):
        super().__init__()
        self.pi = mlp(SD, NA); self.q1, self.q2 = mlp(SD, NA), mlp(SD, NA)
        self.t1, self.t2 = copy.deepcopy(self.q1), copy.deepcopy(self.q2)
        self.opi = optim.Adam(self.pi.parameters(), 3e-4)
        self.oq = optim.Adam(list(self.q1.parameters()) + list(self.q2.parameters()), 1e-3)
        self.alpha, self.tau = alpha, tau

    def _dist(self, x, m):
        lg = self.pi(x).masked_fill(~m, -1e9)
        logp = F.log_softmax(lg, -1); p = logp.exp()
        return p, logp.masked_fill(~m, 0.0)

    def act(self, x, m):
        with torch.no_grad():
            p, _ = self._dist(torch.from_numpy(x).unsqueeze(0), torch.from_numpy(m).unsqueeze(0))
        p = p[0].numpy().astype(np.float64)
        if self.training:
            p = p / p.sum(); return int(np.random.choice(NA, p=p))
        return int(p.argmax())

    def _update(self, X, A, R, X2, M2, D):
        with torch.no_grad():
            p2, lp2 = self._dist(X2, M2)
            qt = torch.min(self.t1(X2), self.t2(X2))
            v = (p2 * (qt - self.alpha * lp2)).sum(1, keepdim=True)
            y = R + GAMMA * (1 - D) * v
        lq = F.mse_loss(self.q1(X).gather(1, A), y) + F.mse_loss(self.q2(X).gather(1, A), y)
        self.oq.zero_grad(); lq.backward(); self.oq.step()
        M = self._mask_from_state(X)      # rebuild action mask from the state flags
        p, lp = self._dist(X, M)
        with torch.no_grad(): qm = torch.min(self.q1(X), self.q2(X))
        la = (p * (self.alpha * lp - qm)).sum(1).mean()
        self.opi.zero_grad(); la.backward(); self.opi.step()
        for q, t in ((self.q1, self.t1), (self.q2, self.t2)):
            for pp, tp in zip(q.parameters(), t.parameters()):
                tp.data.mul_(1 - self.tau).add_(self.tau * pp.data)

    @staticmethod
    def _mask_from_state(X):
        # state layout: x[6]=hop/HMAX (hop 0 -> local valid), x[5]=in RSU range, cand valid flags
        m = torch.zeros(X.shape[0], NA, dtype=torch.bool)
        m[:, 0] = X[:, 6] < 1e-6
        m[:, 1] = X[:, 5] > 0.5
        for c in range(NC): m[:, 2 + c] = X[:, 8 + 6 * c + 4] > 0.5
        return m


class A3CNet(nn.Module):
    def __init__(self):
        super().__init__(); self.pi = mlp(SD, NA); self.v = mlp(SD, 1)


class A3CWorker(Policy):
    """Asynchronous advantage actor-critic: local net, gradients pushed to a shared global net."""
    def __init__(self, gnet, gopt, lock):
        self.g, self.gopt, self.lock = gnet, gopt, lock
        self.net = A3CNet(); self.sync()

    def sync(self): self.net.load_state_dict(self.g.state_dict())

    def act(self, x, m):
        with torch.no_grad():
            lg = self.net.pi(torch.from_numpy(x)).masked_fill(~torch.from_numpy(m), -1e9)
            p = F.softmax(lg, -1).numpy().astype(np.float64)
        if self.training: p /= p.sum(); return int(np.random.choice(NA, p=p))
        return int(p.argmax())

    def learn(self, trajs):
        X, A, G, M = [], [], [], []
        for tr in trajs:
            g = 0.0; rets = []
            for e in reversed(tr): g = e["r"] + GAMMA * g; rets.append(g)
            rets.reverse()
            for e, gg in zip(tr, rets): X.append(e["x"]); A.append(e["a"]); G.append(gg); M.append(e["m"])
        if not X: return
        X = torch.from_numpy(np.stack(X)); A = torch.tensor(A).unsqueeze(1)
        G = torch.tensor(G, dtype=torch.float32).unsqueeze(1); M = torch.from_numpy(np.stack(M))
        lg = self.net.pi(X).masked_fill(~M, -1e9); logp = F.log_softmax(lg, -1)
        v = self.net.v(X); adv = G - v
        ent = -(logp.exp() * logp.masked_fill(~M, 0.0)).sum(1).mean()
        loss = -(logp.gather(1, A) * adv.detach()).mean() + 0.5 * adv.pow(2).mean() - 0.01 * ent
        self.net.zero_grad(); loss.backward(); nn.utils.clip_grad_norm_(self.net.parameters(), 5.0)
        with self.lock:
            for gp, lp in zip(self.g.parameters(), self.net.parameters()): gp.grad = lp.grad.clone()
            self.gopt.step(); self.sync()


# ----------------------------------------------------------------------------------
# VEC environment + simulator
# ----------------------------------------------------------------------------------
class Sim:
    def __init__(self, nv, mal, seed, method, policy):
        C = CFG
        self.rng = np.random.default_rng(seed); r = self.rng
        self.nv, self.method, self.policy = nv, method, policy
        self.proposed = method == "SAC+DGSO"
        self.pos = r.uniform(0, C["area"], (nv, 2))
        sp = r.uniform(*C["v_range"], nv) / 3.6; h = r.uniform(0, 2 * np.pi, nv)
        self.vel = np.stack([sp * np.cos(h), sp * np.sin(h)], 1)
        self.batt = r.uniform(0.15, 1.0, nv); self.f = r.uniform(*C["local_f"], nv)
        self.pt = r.uniform(*C["pt"], nv)
        self.mal = np.zeros(nv, bool); self.mal[r.choice(nv, int(round(mal * nv)), replace=False)] = True
        self.rsu = np.array([[500, 500], [1500, 500], [500, 1500], [1500, 1500]], float)
        self.trust = np.full((nv, nv), 0.7)
        NR, K = C["n_rsu"], C["k_srv"]
        self.srv_f = r.uniform(*C["edge_f"], (NR, K)); self.busy = np.zeros((NR, K)); self.work = np.zeros((NR, K))
        iters = C["dgso_iters_train"] if getattr(policy, "training", False) else C["dgso_iters"]
        self.dgso = DGSO(C["dgso_N"], iters, C["dgso_rp"], C["dgso_w"], True,
                         np.random.default_rng(seed + 7)) if self.proposed else None
        self.log, self.pending, self.buf = [], [[] for _ in range(NR)], []
        self.stats = defaultdict(float); self.refresh()

    # ---- geometry / mobility
    def refresh(self):
        self.D = np.linalg.norm(self.pos[:, None, :] - self.pos[None, :, :], axis=2)
        self.Drsu = np.linalg.norm(self.pos[:, None, :] - self.rsu[None], axis=2)
        self.nr, self.dr = self.Drsu.argmin(1), self.Drsu.min(1)

    def move(self, dt):
        A = CFG["area"]; self.pos += self.vel * dt
        for ax in (0, 1):
            lo, hi = self.pos[:, ax] < 0, self.pos[:, ax] > A
            self.vel[lo | hi, ax] *= -1; self.pos[:, ax] = np.clip(self.pos[:, ax], 0, A)
        self.refresh()

    def lifetime(self, i, j):
        v = np.linalg.norm(self.vel[i] - self.vel[j])
        return min((CFG["v2v_range"] - self.D[i, j]) / max(v, 0.05), 600.0)

    # ---- trust (EWMA direct trust + indirect recommendations + RSU weighted voting)
    def resolve_trust(self, i, j):
        d = self.trust[i, j]
        nb = np.where((self.D[j] <= CFG["v2v_range"]) & (self.D[j] > 0))[0]; nb = nb[nb != i]
        if len(nb) == 0: return d
        if len(nb) > 5: nb = self.rng.choice(nb, 5, replace=False)
        # malicious recommenders: boost fellow attackers, bad-mouth honest nodes
        rep = np.where(self.mal[nb], 0.95 if self.mal[j] else 0.1, self.trust[nb, j])
        if abs(d - rep.mean()) > CFG["conflict_thr"]:
            w = np.maximum(self.trust[i, nb], 0.05)
            vote = (w * rep).sum() / w.sum()
            return CFG["beta"] * d + (1 - CFG["beta"]) * vote
        return d

    # ---- state / mask
    def build(self, cur, task, hops, t, visited):
        C = CFG; R = C["v2v_range"]
        nb = np.where((self.D[cur] <= R) & (self.D[cur] > 0))[0]
        prog = self.dr[cur] - self.dr[nb]; sel = prog > 10
        nb, prog = nb[sel], prog[sel]; o = np.argsort(-prog)[:8]
        infos = []
        for j, p in zip(nb[o], prog[o]):
            j = int(j)
            if j in visited: continue
            d = self.D[cur, j]; rt = link_rate(self.pt[cur], pl_v2v(d))
            lt = self.lifetime(cur, j); tt = task["bits"] / rt; T = 0.0
            if self.proposed:
                T = self.resolve_trust(cur, j)
                ok = (T >= C["tmin"]) and (lt >= C["lmin"])
                if self.mal[j]:
                    self.stats["mal_seen"] += 1
                    if not ok: self.stats["mal_filt"] += 1
                if not ok: continue                          # Candidate_set / action-mask filter
            infos.append(dict(j=j, p=float(p), rt=rt, lt=lt, tt=tt, T=T))
            if len(infos) == NC: break
        d = self.dr[cur]; inr = d <= C["v2i_range"]
        if inr:
            rt_i = link_rate(self.pt[cur], pl_v2i(d))
            lt_i = (C["v2i_range"] - d) / max(np.linalg.norm(self.vel[cur]), 0.05)
            ratio_i = min(task["bits"] / rt_i / lt_i, 2) / 2
        else: ratio_i = 1.0
        x = np.zeros(SD, np.float32)
        x[:8] = [np.log10(task["cyc"]) / 11, np.log10(task["size"]) / 3, np.clip((task["dl"] - t) / 8, -1, 1),
                 self.batt[cur], d / 1500, float(inr), hops / C["hmax"], ratio_i]
        for c, f in enumerate(infos):
            x[8 + 6 * c: 14 + 6 * c] = [f["p"] / 250, min(f["tt"] / f["lt"], 2) / 2, min(f["lt"], 30) / 30,
                                       f["T"], 1.0, min(f["rt"] / 1e8, 1)]
        m = np.zeros(NA, bool); m[0] = hops == 0; m[1] = inr; m[2:2 + len(infos)] = True
        return x, m, infos

    # ---- hop-by-hop forwarding of one encrypted task
    def route(self, i, task):
        C = CFG; cur, path, hops, visited = i, [i], 0, {i}
        t = task["t0"] + task["bits"] / 8 / C["ascon_veh"]        # Ascon encryption at vehicle
        E, traj, st = 0.0, [], "fail"
        while True:
            x, m, infos = self.build(cur, task, hops, t, visited)
            if not m.any(): st = "none"; break
            a = self.policy.act(x, m)
            traj.append(dict(x=x, a=a, m=m, r=-0.05))
            if a == 0: st = "local"; break
            if a == 1:                                                       # direct to RSU
                d = self.dr[cur]; tt = task["bits"] / link_rate(self.pt[cur], pl_v2i(d))
                lt = (C["v2i_range"] - d) / max(np.linalg.norm(self.vel[cur]), 0.05)
                E += self.pt[cur] * tt; t += tt; traj[-1]["r"] -= 0.2 * min(tt, 5)
                if tt > lt: st = "break"; break
                hops += 1
                if self.rng.random() < C["loss_p"]: st = "loss"; break
                st = "rsu"; break
            c = infos[a - 2]; j, tt = c["j"], c["tt"]                        # forward to a vehicle
            E += (self.pt[cur] + C["p_rx"]) * tt; t += tt; traj[-1]["r"] -= 0.2 * min(tt, 5)
            if tt > c["lt"]: st = "break"; break
            hops += 1
            if self.mal[j] and self.rng.random() < C["drop_p"]:               # blackhole
                self.stats["bh_drop"] += 1
                if self.proposed: self.trust[cur, j] = C["alpha"] * self.trust[cur, j]
                st = "drop"; break
            if self.rng.random() < C["loss_p"]: st = "loss"; break
            if self.proposed:
                self.trust[cur, j] = C["alpha"] * self.trust[cur, j] + (1 - C["alpha"]) * 1.0
            path.append(j); visited.add(j); cur = j
            if hops >= C["hmax"]: st = "hops"; break
        return dict(st=st, path=path, t=t, E=E, traj=traj, hops=hops, rsu=int(self.nr[path[-1]]))

    def finish(self, traj, succ, E):
        if traj:
            traj[-1]["r"] += (5.0 if succ else -5.0) - 0.01 * E
            self.buf.append(traj)

    def handle_task(self, i, t0):
        C = CFG; r = self.rng
        size = float(np.clip(r.lognormal(C["size_mu"], C["size_sigma"]), C["size_lo"], C["size_hi"]))
        byt = size * 1e6
        task = dict(i=i, t0=t0, size=size, bits=byt * 8 + 128, cyc=C["cpb"] * byt,
                    dl=t0 + r.uniform(C["dl_lo"], C["dl_hi"]) + C["dl_per_mb"] * size)
        rt = self.route(i, task)
        rec = dict(off=rt["st"] != "local", deliv=False, succ=False, e2e=None, hops=rt["hops"],
                   proc=None, E=rt["E"], bits=task["bits"])
        if rt["st"] == "local":
            tf = t0 + task["cyc"] / self.f[i]; E = C["kappa"] * self.f[i] ** 2 * task["cyc"]
            rec.update(E=E, e2e=tf - t0, succ=tf <= task["dl"], hops=0)
            self.finish(rt["traj"], rec["succ"], E)
        elif rt["st"] == "rsu":
            rec["deliv"] = True
            self.pending[rt["rsu"]].append(dict(task=task, r=rt, rec=rec))
        else:
            self.finish(rt["traj"], False, rt["E"])
        self.log.append(rec)

    def return_path(self, path, r, res_bits):
        C = CFG; last = path[-1]
        d = min(self.Drsu[last, r], C["v2i_range"])
        rate = link_rate(C["pt_rsu"], pl_v2i(d)); t, ok = res_bits / rate, True
        for idx in range(len(path) - 1, 0, -1):
            a, b = path[idx], path[idx - 1]
            if self.mal[a] and self.rng.random() < C["drop_p"]: ok = False; break
            rate = link_rate(self.pt[a], pl_v2v(min(self.D[a, b], C["v2v_range"]))); t += res_bits / rate
        return t, ok, C["p_rx"] * res_bits / rate

    def schedule_all(self, tnow):
        C = CFG; K = C["k_srv"]
        for r in range(C["n_rsu"]):
            P = self.pending[r]
            if not P: continue
            self.pending[r] = []
            P.sort(key=lambda p: p["task"]["dl"])                           # EDF order
            T = [p["task"] for p in P]; M = len(T)
            cyc = np.array([x["cyc"] for x in T]); dl = np.array([x["dl"] for x in T])
            bits = np.array([x["bits"] for x in T])
            ra = np.maximum(np.array([p["r"]["t"] for p in P]), tnow)
            f, busy = self.srv_f[r], self.busy[r]
            ex = cyc[:, None] / f[None, :] + (bits / 8 / C["ascon_srv"])[:, None]   # compute + Ascon decrypt
            prox = np.array([x["i"] % K for x in T])                         # proximal node (header parsing only)
            xfer = (bits / C["backhaul"])[:, None] * (np.arange(K)[None, :] != prox[:, None])
            rdy = ra[:, None] + xfer
            c = dict(ex=ex, rdy=rdy, busy=busy.copy(), dl=dl, ra=ra, t0=float(ra.min()),
                     eE=C["kappa_edge"] * f[None, :] ** 2 * cyc[:, None], dref=float((dl - ra).mean()) + 1e-6)
            assign = self.dgso.optimize(c, r) if self.proposed else least_loaded(c)
            fins = np.zeros(M)
            for m in range(M):                                              # execute in EDF order per server
                k = int(assign[m]); st = max(busy[k], rdy[m, k]); fins[m] = st + ex[m, k]
                busy[k] = fins[m]; self.work[r, k] += ex[m, k]
            for m, p in enumerate(P):
                task, rt, rec = p["task"], p["r"], p["rec"]
                res_bits = max(0.02 * task["size"], 0.05) * 8e6
                rd, ok, Erx = self.return_path(rt["path"], r, res_bits)
                comp = fins[m] + res_bits / C["backhaul"] + rd + res_bits / 8 / C["ascon_veh"]
                succ = ok and comp <= task["dl"]
                rec.update(succ=succ, e2e=(comp - task["t0"]) if ok else None,
                           proc=fins[m] - rt["t"], E=rec["E"] + Erx)
                self.finish(rt["traj"], succ, rec["E"])
        if getattr(self.policy, "training", False): self.policy.learn(self.buf)
        self.buf = []

    def run(self):
        C = CFG; T = C["steps"]
        for k in range(T):
            gen = np.where(self.rng.random(self.nv) < C["p_gen"])[0]
            for i in gen: self.handle_task(int(i), k * C["dt"] + self.rng.random() * C["dt"])
            if (k + 1) % C["sched_every"] == 0 or k == T - 1: self.schedule_all((k + 1) * C["dt"])
            self.move(C["dt"])
        return self.metrics()

    def metrics(self):
        L = self.log; n = max(len(L), 1); Tsim = CFG["steps"] * CFG["dt"]
        off = [x for x in L if x["off"]]; dv = [x for x in off if x["deliv"]]
        done = [x for x in L if x["e2e"] is not None]; nan = float("nan")
        util = self.work / Tsim
        return dict(
            delay_ms=1e3 * np.mean([x["e2e"] for x in done]) if done else nan,
            pdr=100 * len(dv) / max(len(off), 1),
            throughput=sum(x["bits"] for x in dv) / 1e6 / Tsim,
            hops=np.mean([x["hops"] for x in dv]) if dv else nan,
            proc_ms=1e3 * np.mean([x["proc"] for x in dv if x["proc"] is not None]) if dv else nan,
            success=100 * sum(x["succ"] for x in L) / n, miss=100 - 100 * sum(x["succ"] for x in L) / n,
            energy=np.mean([x["E"] for x in L]) if L else nan,
            detect=100 * self.stats["mal_filt"] / self.stats["mal_seen"] if self.stats["mal_seen"] else 0.0,
            drop=100 * self.stats["bh_drop"] / max(len(off), 1),
            imbalance=float(util.std()), util=100 * float(np.minimum(util, 1).mean()),
            offload_ratio=100 * len(off) / n, n_tasks=len(L))


# ----------------------------------------------------------------------------------
# Training / evaluation
# ----------------------------------------------------------------------------------
TRAIN_NV = [200, 300, 400]


def train_a3c(episodes, workers=4):
    g = A3CNet(); opt = optim.Adam(g.parameters(), 5e-4); lock = threading.Lock(); curve = []

    def work(w):
        pol = A3CWorker(g, opt, lock); pol.training = True
        for ep in range(max(episodes // workers, 1)):
            met = Sim(TRAIN_NV[(ep + w) % 3], 0.15, 20000 + w * 1000 + ep, "A3C", pol).run()
            curve.append(met["success"])
    th = [threading.Thread(target=work, args=(w,)) for w in range(workers)]
    [t.start() for t in th]; [t.join() for t in th]
    ev = A3CWorker(g, opt, lock); ev.training = False
    return ev, curve


def train_all(episodes):
    pols, curves = {}, {}
    for name in METHODS:
        t0 = time.time()
        if name == "A3C":
            pols[name], curves[name] = train_a3c(episodes)
        else:
            pol = SACAgent() if name == "SAC+DGSO" else DQNAgent(double=(name == "DDQN"))
            pol.training = True; curve = []
            for ep in range(episodes):
                curve.append(Sim(TRAIN_NV[ep % 3], 0.15, 1000 + ep, name, pol).run()["success"])
                pol.end_episode()
            pol.training = False; pols[name], curves[name] = pol, curve
        print(f"  trained {name:9s} {episodes} episodes in {time.time() - t0:.0f}s "
              f"(last-10 train success {np.mean(curves[name][-10:]):.1f}%)", flush=True)
    return pols, curves


KEYS = ["delay_ms", "pdr", "throughput", "hops", "proc_ms", "success", "miss",
        "energy", "detect", "drop", "imbalance", "util"]
TITLES = {"delay_ms": ("End-to-end delay", "ms"), "pdr": ("Packet delivery ratio", "%"),
          "throughput": ("Throughput", "Mbps"), "hops": ("Average hop count", "hops"),
          "proc_ms": ("Task processing time", "ms"), "success": ("Task success rate", "%"),
          "miss": ("Deadline miss rate", "%"), "energy": ("Energy per task", "J"),
          "detect": ("Attack detection rate", "%"), "drop": ("Malicious task drop rate", "%"),
          "imbalance": ("Load imbalance (std of utilisation)", "std"), "util": ("Edge server utilisation", "%")}
STYLE = {"SAC+DGSO": ("tab:red", "o", "-"), "DQN": ("tab:blue", "s", "--"),
         "DDQN": ("tab:green", "^", "-."), "A3C": ("tab:purple", "d", ":")}


def sweep(pols, xs, seeds, mode):
    res = {m: {k: np.zeros((len(xs), seeds)) for k in KEYS} for m in METHODS}
    for xi, x in enumerate(xs):
        nv, mal = (x, 0.15) if mode == "nv" else (300, x)
        for s in range(seeds):
            for m in METHODS:
                met = Sim(nv, mal, 90000 + 17 * s + xi, m, pols[m]).run()
                for k in KEYS: res[m][k][xi, s] = met[k]
        print(f"  [{mode}] x={x} done", flush=True)
    return res


def plot_sweep(res, xs, xlabel, outdir, prefix):
    xs_plot = [100 * x for x in xs] if prefix == "mal" else xs
    def draw(ax, k):
        for m in METHODS:
            a = res[m][k]; mu = np.nanmean(a, 1)
            ci = 1.96 * np.nanstd(a, 1, ddof=1) / math.sqrt(a.shape[1]) if a.shape[1] > 1 else 0 * mu
            c, mk, ls = STYLE[m]
            ax.errorbar(xs_plot, mu, yerr=ci, color=c, marker=mk, ls=ls, capsize=2, label=m, lw=1.6, ms=5)
        ax.set_xlabel(xlabel); ax.set_ylabel(f"{TITLES[k][0]} ({TITLES[k][1]})")
        ax.grid(alpha=.3); ax.legend(fontsize=7)
    fig, axs = plt.subplots(3, 4, figsize=(20, 12))
    for ax, k in zip(axs.flat, KEYS): draw(ax, k); ax.set_title(TITLES[k][0], fontsize=10)
    fig.tight_layout(); fig.savefig(os.path.join(outdir, f"{prefix}_all.png"), dpi=130); plt.close(fig)
    for k in KEYS:
        fig, ax = plt.subplots(figsize=(5.5, 4)); draw(ax, k); fig.tight_layout()
        fig.savefig(os.path.join(outdir, f"{prefix}_{k}.png"), dpi=150); plt.close(fig)


# ----------------------------------------------------------------------------------
# DGSO optimisation-behaviour study  (metrics 13-16)
# ----------------------------------------------------------------------------------
def synth_ctx(rng, M=40, K=4, base=None, keep=0.7):
    C = CFG
    if base is None:
        cyc = np.clip(rng.lognormal(math.log(6e9), 1.0, M), 1e8, 2e11); ra = rng.uniform(0, 1, M)
        dl = ra + rng.uniform(1.5, 6, M) + 3e-11 * cyc
        f = rng.uniform(*C["edge_f"], K); busy = rng.uniform(0, 1.5, K)
    else:
        cyc, ra, dl, f, busy = (b.copy() for b in base); n = int(M * (1 - keep))
        cyc[-n:] = np.clip(rng.lognormal(math.log(6e9), 1.0, n), 1e8, 2e11)
        ra[-n:] = rng.uniform(0, 1, n); dl[-n:] = ra[-n:] + rng.uniform(1.5, 6, n) + 3e-11 * cyc[-n:]
    bits = cyc / C["cpb"] * 8
    xf = (bits / C["backhaul"])[:, None] * (rng.random((M, K)) < 0.75)
    c = dict(ex=cyc[:, None] / f[None] + 0.01, rdy=ra[:, None] + xf, busy=busy.copy(), dl=dl, ra=ra, t0=0.0,
             eE=C["kappa_edge"] * f[None] ** 2 * cyc[:, None], dref=float((dl - ra).mean()))
    return c, (cyc, ra, dl, f, busy)


def dgso_study(outdir, quick):
    C = CFG; rng = np.random.default_rng(3)
    iters = 60 if quick else C["dgso_iters"]; nw = 4 if quick else 10
    def fit_of(a, c):
        d = DGSO(1, 1, w=C["dgso_w"], rng=np.random.default_rng(0)); _, _, Fv, _ = d._eval(np.eye(c["ex"].shape[1])[a][None] * 0.8 + 0.1, c)
        return float(Fv[0])
    # (a) fitness curve vs random / greedy baselines
    c, _ = synth_ctx(rng); d = DGSO(C["dgso_N"], iters, rng=np.random.default_rng(1), warm=False); d.optimize(c)
    rnd = np.mean([fit_of(rng.integers(0, 4, 40), c) for _ in range(200)]); gr = fit_of(greedy_eft(c), c)
    ll = fit_of(least_loaded(c), c); curve = d.hist
    # (b) population size
    pops = [10, 20, 30, 50, 100]; popF = []
    for N in pops:
        v = []
        for w in range(nw):
            cc, _ = synth_ctx(rng); dd = DGSO(N, iters, rng=np.random.default_rng(w), warm=False); dd.optimize(cc); v.append(dd.hist[-1])
        popF.append((np.mean(v), np.std(v)))
    # (c) warm vs cold over consecutive windows (70 % of tasks carried over)
    out = {}
    for warm in (True, False):
        dd = DGSO(C["dgso_N"], iters, rng=np.random.default_rng(5), warm=warm); r2 = np.random.default_rng(11)
        base = None; curves, its = [], []
        for w in range(nw + 1):
            cc, base = synth_ctx(r2, base=base); dd.optimize(cc, key=0)
            h = dd.hist
            if w > 0:
                curves.append(h); its.append(int(np.argmax(h <= h[-1] * 1.01)) + 1)
        out[warm] = (np.mean(curves, 0), its, [h[-1] for h in curves])
    fig, axs = plt.subplots(1, 4, figsize=(20, 4))
    axs[0].plot(curve, color="tab:red", label="DGSO best fitness"); axs[0].axhline(rnd, color="gray", ls="--", label="random")
    axs[0].axhline(ll, color="tab:blue", ls=":", label="least-loaded"); axs[0].axhline(gr, color="tab:green", ls="-.", label="greedy EFT")
    axs[0].set_xlabel("iteration"); axs[0].set_ylabel("F_best"); axs[0].set_title("Fitness progression"); axs[0].legend(); axs[0].grid(alpha=.3)
    axs[1].errorbar(pops, [p[0] for p in popF], yerr=[p[1] for p in popF], marker="o", capsize=3, color="tab:red")
    axs[1].set_xlabel("population size N"); axs[1].set_ylabel("final F_best"); axs[1].set_title("Solution quality vs N"); axs[1].grid(alpha=.3)
    for warm, col in ((True, "tab:red"), (False, "tab:blue")):
        axs[2].plot(out[warm][0], color=col, label="warm start" if warm else "cold start")
    axs[2].set_xlabel("iteration"); axs[2].set_ylabel("F_best (mean over windows)"); axs[2].set_title("Warm vs cold start"); axs[2].legend(); axs[2].grid(alpha=.3)
    axs[3].bar(["warm", "cold"], [np.mean(out[True][1]), np.mean(out[False][1])], yerr=[np.std(out[True][1]), np.std(out[False][1])],
               color=["tab:red", "tab:blue"], capsize=4)
    axs[3].set_ylabel("iterations to within 1 % of final F"); axs[3].set_title("Convergence speed")
    fig.tight_layout(); fig.savefig(os.path.join(outdir, "dgso_study.png"), dpi=140); plt.close(fig)
    summary = dict(random=rnd, least_loaded=ll, greedy=gr, dgso_final=float(curve[-1]),
                   pop={str(n): p[0] for n, p in zip(pops, popF)},
                   iters_warm=float(np.mean(out[True][1])), iters_cold=float(np.mean(out[False][1])),
                   final_warm=float(np.mean(out[True][2])), final_cold=float(np.mean(out[False][2])))
    json.dump(summary, open(os.path.join(outdir, "dgso_study.json"), "w"), indent=2)
    print("  DGSO study:", json.dumps(summary, indent=1), flush=True)


# ----------------------------------------------------------------------------------
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--exp", default="all", choices=["all", "sweep", "mal", "dgso"])
    ap.add_argument("--episodes", type=int, default=150)
    ap.add_argument("--seeds", type=int, default=5)
    ap.add_argument("--out", default="results")
    ap.add_argument("--quick", action="store_true")
    ap.add_argument("--smoke", action="store_true")
    a = ap.parse_args(); os.makedirs(a.out, exist_ok=True)
    nvs, mals = [100, 200, 300, 400, 500, 600], [0.10, 0.15, 0.20, 0.25]
    if a.quick: a.episodes, a.seeds, nvs = 60, 3, [100, 300, 600]; CFG["dgso_iters"] = 80
    if a.smoke:
        a.episodes, a.seeds, nvs, mals = 4, 1, [100, 200], [0.10, 0.25]
        CFG.update(steps=16, dgso_iters=15, dgso_iters_train=8)
    if a.exp in ("all", "dgso"): print("DGSO study..."); dgso_study(a.out, a.quick or a.smoke)
    if a.exp == "dgso": return
    print("Training agents..."); pols, curves = train_all(a.episodes)
    fig, ax = plt.subplots(figsize=(6, 4))
    for m in METHODS:
        c = np.array(curves[m], float); k = max(len(c) // 15, 1)
        ax.plot(np.convolve(c, np.ones(k) / k, "valid") if len(c) > k else c, label=m, color=STYLE[m][0])
    ax.set_xlabel("training episode"); ax.set_ylabel("task success rate (%)"); ax.legend(); ax.grid(alpha=.3)
    fig.tight_layout(); fig.savefig(os.path.join(a.out, "training_curves.png"), dpi=140); plt.close(fig)
    if a.exp in ("all", "sweep"):
        print("Vehicle-density sweep (15 % malicious)...")
        res = sweep(pols, nvs, a.seeds, "nv")
        json.dump({m: {k: v.tolist() for k, v in d.items()} for m, d in res.items()}, open(os.path.join(a.out, "sweep_nv.json"), "w"))
        plot_sweep(res, nvs, "Number of vehicles", a.out, "nv")
    if a.exp in ("all", "mal"):
        print("Malicious-rate sweep (300 vehicles)...")
        res = sweep(pols, mals, a.seeds, "mal")
        json.dump({m: {k: v.tolist() for k, v in d.items()} for m, d in res.items()}, open(os.path.join(a.out, "sweep_mal.json"), "w"))
        plot_sweep(res, mals, "Malicious vehicles (%)", a.out, "mal")
    print("Done. Results in", a.out)


if __name__ == "__main__":
    main()
