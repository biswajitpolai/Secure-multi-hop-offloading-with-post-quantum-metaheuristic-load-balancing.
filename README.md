# Secure Multi-Hop Task Offloading with Post-Quantum Security and Metaheuristic Load Balancing

Python simulation of a secure, trust-aware task-offloading framework for Vehicular Edge Computing (VEC).
It compares the proposed **SAC + DGSO** scheme against **DQN**, **DDQN** and **A3C** baselines under
blackhole attacks and varying vehicle density.


---

## 1. Framework overview

| Step | What happens | In the simulator |
|---|---|---|
| 0 | Edge server ML-KEM-512 key provisioning (offline) | Not executed; assumed done |
| 1 | Vehicle generates a general computational task | Yes (size, CPU cycles, deadline) |
| 2 | Offload decision (LOCAL vs OFFLOAD) | Yes, first action of the RL policy |
| 3 | ML-KEM-512 session key + Ascon-AEAD128 encryption | **Modelled as a processing delay only** (see below) |
| 4 | Trust-aware hop-by-hop V2V forwarding to the RSU | Yes (EWMA trust, conflict resolution, masked policy) |
| 5 | RSU routes the encrypted task to the proximal edge node (header parsing only) | Yes |
| 6 | DGSO scheduling and load balancing on header metadata (cpu_cycles, deadline, data_size) | Yes |
| 7 | Decryption and execution at the **executing** edge server | Decryption modelled as a delay |
| 8 | Result encrypted at the executing server, returned via proximal node, RSU and V2V path | Yes (delay, blackhole drops on return) |

The cryptography itself (ML-KEM-512, Ascon-AEAD128) is **not implemented** in this simulator. Only its
cost is included: Ascon throughput is applied as a per-task delay at the vehicle and at the edge server
(`ascon_veh`, `ascon_srv` in `CFG`). Use a crypto library separately if you need to benchmark the real primitives.

## 2. Methods compared

| Method | Offload / next-hop policy | Trust mask | Edge scheduler |
|---|---|---|---|
| **SAC+DGSO** (proposed) | Discrete SAC with action masking | Yes (T_min, L_min, conflict resolution) | DGSO |
| DQN | DQN, replay buffer, target network | No | Least-loaded server |
| DDQN | Double DQN | No | Least-loaded server |
| A3C | Asynchronous actor-critic (4 threads, shared global network) | No | Least-loaded server |

All four agents are really trained in the same environment with the same state, actions and reward.
The baselines' least-loaded scheduler is an assumption; change `least_loaded()` if your paper specifies another one.

**Policy interface.** Each decision is made per hop. The action space has 6 actions:
`0` = execute locally (first hop only), `1` = send directly to the RSU (if in range),
`2-5` = forward to one of the 4 best next-hop candidates. Invalid actions are masked.
For the proposed method the mask also removes candidates with trust below `T_min` or link lifetime below `L_min`.

## 3. Requirements and installation

- Python 3.9+
- `numpy`, `torch` (CPU is enough), `matplotlib`

```bash
pip install numpy torch matplotlib
```

No GPU is needed. The networks are tiny and most of the time is spent in the numpy simulation.

## 4. Running

```bash
python vec_sim.py --smoke            # ~1-2 min, checks that everything runs
python vec_sim.py --quick            # 60 training episodes, 3 vehicle counts, 3 seeds
python vec_sim.py                    # full run: 150 episodes, 6 vehicle counts, 5 seeds
python vec_sim.py --exp dgso         # only the DGSO optimisation study
python vec_sim.py --out myfolder     # choose the output folder (default: results)
```

| Flag | Meaning |
|---|---|
| `--exp {all,sweep,mal,dgso}` | `sweep` = vehicle-density sweep, `mal` = malicious-rate sweep, `dgso` = DGSO study only |
| `--episodes N` | Training episodes per method (default 150) |
| `--seeds N` | Evaluation seeds per point (default 5) |
| `--quick` | Reduced run (60 episodes, 3 seeds, vehicles 100/300/600, 80 DGSO iterations) |
| `--smoke` | Tiny run for testing only; results are not meaningful |
| `--out DIR` | Output directory |

Reference timings on a Windows laptop (CPU only): the `--quick` run trained all four agents in about
9 minutes; the full run trained them in about 20 minutes (SAC+DGSO is the slowest because DGSO runs in every scheduling window).
The sweeps add further time on top of training.

Note: the agents are retrained on every invocation. Run `--exp all` (the default) to produce all plots in one go.

## 5. Output files

All files are written to the output folder.

| File | Content |
|---|---|
| `nv_all.png`, `nv_<metric>.png` | 12 metrics vs number of vehicles (100-600), 15% malicious |
| `mal_all.png`, `mal_<metric>.png` | 12 metrics vs malicious rate (10-25%), 300 vehicles |
| `training_curves.png` | Task success rate per training episode for each method |
| `dgso_study.png`, `dgso_study.json` | Fitness progression, population-size sensitivity, warm vs cold start, convergence speed |
| `sweep_nv.json`, `sweep_mal.json` | Raw per-seed metric values for every method and x-value |

Plots show the mean over seeds with 95% confidence intervals.

## 6. Performance metrics

| Key | Metric | Better |
|---|---|---|
| `delay_ms` | End-to-end delay (task generation to result received) | Lower |
| `pdr` | Packet delivery ratio (offloaded tasks delivered to the RSU) | Higher |
| `throughput` | Data delivered to the edge per second (Mbps) | Higher |
| `hops` | Average V2V hops per delivered task | Lower |
| `proc_ms` | Processing time at the edge (queue + execution), from RSU arrival | Lower |
| `success` | Tasks completed within the deadline and received correctly | Higher |
| `miss` | Deadline miss rate (100 - success) | Lower |
| `energy` | Average vehicle energy per task (J): transmit, receive, local compute | Lower |
| `detect` | Share of malicious next-hop candidates filtered by the trust mask (proposed method only; 0 for baselines) | Higher |
| `drop` | Tasks dropped by blackhole nodes, as a % of offloaded tasks | Lower |
| `imbalance` | Standard deviation of edge-server utilisation | Lower |
| `util` | Mean edge-server utilisation (%) | Balanced |
| `offload_ratio` | % of tasks offloaded (saved in the JSON only, not plotted) | n/a |

**DGSO study metrics:** convergence speed (iterations to within 1% of final fitness), fitness progression,
solution quality vs population size N, and warm-start benefit.

## 7. Simulation model

**Network and mobility**
- 2000 x 2000 m area, 100-600 vehicles at 10-60 km/h, random headings with boundary reflection.
- 4 RSUs at (500,500), (1500,500), (500,1500), (1500,1500), each with 4 edge servers (10-25 GHz).
- V2V range 250 m, V2I range 500 m, 10 MHz channel, noise -100 dBm.
- V2V path loss: log-distance, eta = 2.7, PL0 = free-space loss at 1 m (5.9 GHz).
  V2I path loss: WINNER II B1 urban, `22.7*log10(d) + 41.0 + 20*log10(f/5)`.
- Rates from the Shannon formula; vehicle TX power 0.1-2.0 W, RSU 2.0 W.
- Edge-to-edge and RSU links: 1 Gbps. Vehicle CPU 1-5 GHz.

**Tasks**
- Size: clipped log-normal between 1 and 1000 MB (median about 8 MB). A uniform 1-1000 MB distribution
  is infeasible with 1.5-6 s deadlines on a 10 MHz channel.
- CPU cycles: 400 cycles per byte. Deadline: U(1.5, 6) s + 0.03 s per MB.
- Each vehicle generates a task with probability 0.05 per 0.5 s step.

**Energy**
- Local computation: `E = kappa * f^2 * C` (kappa = 1e-27). The commonly used form is f^2, not f^3.
- Transmission: `P_tx * t_tx`. Reception: `P_rx * t_rx`, with P_rx = 0.1 W.

**Attack and trust**
- Blackhole attackers (10-25% of vehicles) drop 90% of the tasks they are asked to relay,
  including result packets on the return path.
- As recommenders, attackers give 0.95 for fellow attackers and 0.1 for honest nodes (bad-mouthing).
- Direct trust: EWMA with alpha = 0.7. Indirect trust: recommendations from up to 5 neighbours of the candidate.
  If direct and indirect trust differ by more than 0.3, an RSU-style trust-weighted vote is used and blended with beta = 0.6.
- Candidates need trust >= 0.6 and link lifetime >= 5 s.

**Scheduling windows**
- Offloaded tasks are scheduled every 1 s per RSU. Tasks execute in EDF order on each server.
- Tasks move between the proximal and executing server over the internal link when they differ.

**DGSO settings** (Table 2): population 50, 200 iterations, RP = 0.3, fitness weights 0.4 delay /
0.3 imbalance / 0.2 deadline miss / 0.1 energy, warm start, differential guidance from 3 random agents,
discrete glide jump with probability A, repair for capacity and deadline violations.
During RL training DGSO uses 25 iterations to save time.

## 8. Code structure (`vec_sim.py`)

| Section | Contents |
|---|---|
| `CFG` | All parameters; edit here to change the experiment |
| `pl_v2v`, `pl_v2i`, `link_rate` | Path loss and channel rate |
| `DGSO` | Vectorised DGSO scheduler; `greedy_eft`, `least_loaded` helper schedulers |
| `DQNAgent`, `SACAgent`, `A3CNet`, `A3CWorker` | RL agents with action masking |
| `Sim` | Environment: mobility, trust, hop-by-hop routing, scheduling, execution, return path, metrics |
| `train_all`, `train_a3c` | Training loops (A3C uses 4 threads) |
| `sweep`, `plot_sweep` | Evaluation sweeps and plotting |
| `dgso_study` | DGSO optimisation-behaviour experiments |

## 9. Customising

- **Longer simulation:** raise `steps` in `CFG` (default 80 steps of 0.5 s = 40 s).
- **More or fewer tasks:** change `p_gen`.
- **Harder or easier tasks:** change `cpb`, `size_mu`, `dl_per_mb`.
- **Attack strength:** change `drop_p` and the malicious-rate list in `main()`.
- **Different baseline scheduler:** edit `least_loaded()` or pass your own function in `Sim.schedule_all`.
- **Reward shaping:** see `Sim.route` (per-hop cost) and `Sim.finish` (+5 / -5 terminal reward minus 0.01 x energy).

## 10. Limitations

- Cryptography is modelled as delay only; ML-KEM-512 and Ascon-AEAD128 are not executed.
- A3C uses Python threads and the GIL, so it is asynchronous in updates but not truly parallel.
- Vehicles move on random headings in an open area, not on a road network or real traces.
- Trust is learned per vehicle from its own forwarding observations; each vehicle must see a drop itself
  (or receive conflicting recommendations) before it excludes an attacker.
- DGSO windows hold only newly arrived tasks, so warm start seeds from the previous window by position.
  The DGSO study carries 70% of tasks across windows, which favours warm start more than the main simulation does.
- Reward weights, task intensity and deadlines were chosen for a workable trade-off, not calibrated to a real deployment.
  Always check the training curves and the `offload_ratio` values before drawing conclusions, and
  report results with multiple seeds and confidence intervals.

## 11. Suggested citation of settings in a paper

Report Table 1 (network) and Table 2 (DGSO) values from `CFG`, state that the baselines share the same
environment and state/action definition, state their scheduler, and describe the blackhole model as in Section 7.
