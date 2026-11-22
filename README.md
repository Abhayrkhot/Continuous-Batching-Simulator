# Continuous-Batching-Simulator

`batchsim` is a discrete-event simulator for LLM serving schedulers: static batching,
Orca-style iteration-level batching, vLLM V0's prefill-first scheduler and vLLM V1's chunked
prefill. Its `vllm` policy reproduces vLLM 0.34's scheduler decision for decision; its
step-time model is fitted to iterations recorded inside a running vLLM engine; and every
prediction below was checked against vLLM serving the Azure production LLM traces on an
NVIDIA H200.

- **Same decisions as vLLM.** Driven side by side with vLLM 0.34's own `Scheduler` and
  `AsyncScheduler` over 1,000 randomized request streams (503,963 scheduling steps, 444
  streams with preemption), `batchsim` schedules the same requests with the same token
  counts at every step: **0 mismatches**.
- **Accurate latency predictions.** Over 34 replays of the Azure conversation and
  code traces (25,619 requests; Qwen2.5-7B and Qwen2.5-32B; two scheduler
  configurations), the median error is **4.7% on TTFT p50, 2.7% on TTFT p99,
  3.2% on TPOT p50, 1.5% on TPOT p99 and 1.2% on end-to-end p99**. The
  step model was fitted only on a synthetic calibration replay; its held-out error is
  0.9–1.1% per iteration.
- **Right about configuration trade-offs.** Where two vLLM configurations were measured on the
  same workload, the simulator ordered them correctly on 27 of 32 latency metrics. At
  28 req/s it predicted that a 2048-token budget cuts TPOT p99 by 63% versus the H200 default
  of 8192 (measured: 63%) and raises TTFT p99 by 326% (measured: 402%).
- **Fast.** One hour of the Azure conversation trace (19,366 requests, 604k engine
  iterations) simulates in 4.2 s on one CPU core, so a capacity search over configurations
  takes seconds rather than GPU-hours.

![Predicted vs measured latency percentiles](docs/parity.png)

## Results

### Validation against vLLM 0.34 on one H200

Each point above is one latency percentile of one replay: the client's measurement of real
vLLM against the simulator's prediction for the same request stream. Accuracy by metric:

| metric | median abs. error | 90th pct abs. error | within 10 % |
|---|---:|---:|---:|
| TTFT P50 | 4.7% | 13.4% | 85% |
| TTFT P90 | 3.9% | 12.2% | 79% |
| TTFT P99 | 2.7% | 12.3% | 79% |
| TPOT P50 | 3.2% | 7.3% | 91% |
| TPOT P90 | 2.8% | 8.9% | 91% |
| TPOT P99 | 1.5% | 9.0% | 94% |
| E2E P50 | 1.6% | 5.4% | 97% |
| E2E P99 | 1.2% | 8.0% | 97% |

Errors stay small until the server saturates, where queueing amplifies a few percent of
step-time error into tens of percent of waiting time:

| band | replays | TTFT p50 | TTFT p99 | TPOT p50 | TPOT p99 | E2E p50 | E2E p99 |
|---|---:|---:|---:|---:|---:|---:|---:|
| below saturation | 16 | 3.4% | 1.8% | 2.5% | 1.1% | 1.4% | 1.1% |
| saturated | 18 | 6.5% | 5.2% | 3.3% | 2.3% | 2.0% | 2.6% |

![Qwen2.5-7B, Azure conversation trace, two token budgets](docs/conv_qwen7b.png)

The Qwen2.5-32B replays are the closest fit: TTFT p50 within 5.2% and TTFT p99 within 6.4%
on all ten, including the code trace at 2 req/s. That trace is bursty enough that the measured
median TTFT is 3,066 ms (predicted: 3,073 ms).

Qwen2.5-7B: fitted on 19,498 calibration iterations; held-out median error 1.1% (mean 1.8%); on 193,299 iterations of the validation replays, median error 2.5%. Frontend latency 3.7 ms, idle launch 1.5 ms.

Qwen2.5-32B: fitted on 8,548 calibration iterations; held-out median error 0.9% (mean 1.4%); on 34,970 iterations of the validation replays, median error 0.7%. Frontend latency 11.4 ms, idle launch 0.0 ms.

<details>
<summary>Every replay (measured / predicted)</summary>

**Qwen2.5-7B, `budget 2048, 256 seqs`** (max_num_batched_tokens=2048, max_num_seqs=256, 128,319 KV blocks)

| trace @ req/s | requests | TTFT p50 (ms) | TTFT p99 (ms) | TPOT p50 (ms) | TPOT p99 (ms) | E2E p99 (s) |
|---|---:|---:|---:|---:|---:|---:|
| code @ 1.0 | 240 | 51 / 52 (+2%) | 321 / 314 (-2%) | 4.8 / 4.9 (+2%) | 20.9 / 20.5 (-2%) | 0.81 / 0.81 (+0%) |
| code @ 2.0 | 300 | 112 / 105 (-6%) | 553 / 535 (-3%) | 6.9 / 7.2 (+3%) | 47.2 / 46.2 (-2%) | 1.33 / 1.32 (-0%) |
| code @ 3.0 | 360 | 422 / 398 (-6%) | 2289 / 2040 (-11%) | 47.6 / 45.6 (-4%) | 51.4 / 47.7 (-7%) | 5.29 / 4.92 (-7%) |
| code @ 4.0 | 480 | 755 / 686 (-9%) | 2402 / 2135 (-11%) | 48.2 / 45.7 (-5%) | 51.8 / 47.8 (-8%) | 5.22 / 4.82 (-8%) |
| code @ 6.0 | 720 | 2083 / 1841 (-12%) | 9064 / 7933 (-12%) | 49.9 / 46.7 (-6%) | 52.2 / 48.3 (-8%) | 11.75 / 10.71 (-9%) |
| conv @ 4.0 | 480 | 31 / 34 (+8%) | 110 / 110 (-0%) | 5.4 / 5.5 (+1%) | 7.7 / 7.7 (-0%) | 3.14 / 3.17 (+1%) |
| conv @ 8.0 | 960 | 33 / 36 (+7%) | 151 / 151 (+0%) | 6.2 / 6.2 (+1%) | 9.6 / 9.5 (-1%) | 3.89 / 3.90 (+0%) |
| conv @ 12.0 | 1440 | 37 / 38 (+4%) | 221 / 200 (-10%) | 7.4 / 7.5 (+1%) | 14.7 / 14.6 (-0%) | 4.82 / 4.80 (-0%) |
| conv @ 16.0 | 1500 | 44 / 44 (-1%) | 304 / 263 (-13%) | 8.9 / 9.2 (+4%) | 20.3 / 19.4 (-4%) | 6.12 / 6.21 (+1%) |
| conv @ 20.0 | 1500 | 67 / 61 (-9%) | 408 / 359 (-12%) | 13.4 / 13.1 (-2%) | 30.3 / 27.3 (-10%) | 9.55 / 9.24 (-3%) |
| conv @ 24.0 | 1500 | 158 / 113 (-29%) | 908 / 594 (-35%) | 29.0 / 25.3 (-13%) | 53.2 / 48.1 (-10%) | 19.67 / 17.66 (-10%) |
| conv @ 28.0 | 1500 | 432 / 340 (-21%) | 4124 / 3387 (-18%) | 36.9 / 35.4 (-4%) | 52.1 / 50.8 (-2%) | 22.17 / 21.04 (-5%) |

**Qwen2.5-7B, `budget 8192, 1024 seqs`** (max_num_batched_tokens=8192, max_num_seqs=1024, 126,399 KV blocks)

| trace @ req/s | requests | TTFT p50 (ms) | TTFT p99 (ms) | TPOT p50 (ms) | TPOT p99 (ms) | E2E p99 (s) |
|---|---:|---:|---:|---:|---:|---:|
| code @ 1.0 | 240 | 49 / 50 (+2%) | 306 / 311 (+2%) | 4.7 / 4.9 (+5%) | 23.0 / 23.4 (+2%) | 0.78 / 0.81 (+4%) |
| code @ 2.0 | 300 | 116 / 124 (+7%) | 529 / 562 (+6%) | 6.7 / 7.0 (+3%) | 90.4 / 92.4 (+2%) | 1.33 / 1.36 (+2%) |
| code @ 3.0 | 360 | 434 / 426 (-2%) | 2109 / 2109 (-0%) | 97.7 / 96.4 (-1%) | 190.3 / 189.0 (-1%) | 6.14 / 6.14 (-0%) |
| code @ 4.0 | 480 | 774 / 770 (-0%) | 2210 / 2209 (-0%) | 91.6 / 92.2 (+1%) | 191.2 / 190.0 (-1%) | 6.17 / 6.17 (-0%) |
| code @ 6.0 | 720 | 1878 / 1865 (-1%) | 7985 / 7865 (-2%) | 185.0 / 185.1 (+0%) | 192.8 / 191.1 (-1%) | 13.77 / 13.65 (-1%) |
| conv @ 4.0 | 480 | 32 / 34 (+8%) | 101 / 101 (+1%) | 5.4 / 5.5 (+2%) | 7.7 / 7.7 (+0%) | 3.13 / 3.17 (+1%) |
| conv @ 8.0 | 960 | 34 / 36 (+6%) | 153 / 155 (+2%) | 6.1 / 6.3 (+3%) | 9.5 / 9.6 (+1%) | 3.81 / 3.92 (+3%) |
| conv @ 12.0 | 1440 | 36 / 38 (+6%) | 221 / 220 (-0%) | 7.2 / 7.5 (+4%) | 14.7 / 14.8 (+1%) | 4.72 / 4.83 (+2%) |
| conv @ 16.0 | 1500 | 43 / 44 (+4%) | 303 / 285 (-6%) | 8.9 / 9.4 (+5%) | 20.5 / 20.4 (-1%) | 6.12 / 6.31 (+3%) |
| conv @ 20.0 | 1500 | 63 / 59 (-7%) | 381 / 350 (-8%) | 13.1 / 13.4 (+3%) | 30.1 / 28.7 (-5%) | 9.36 / 9.44 (+1%) |
| conv @ 24.0 | 1500 | 117 / 101 (-14%) | 571 / 535 (-6%) | 28.7 / 25.4 (-12%) | 65.1 / 57.5 (-12%) | 19.89 / 18.10 (-9%) |
| conv @ 28.0 | 1500 | 230 / 197 (-14%) | 822 / 796 (-3%) | 51.2 / 49.5 (-3%) | 141.1 / 137.9 (-2%) | 31.86 / 31.59 (-1%) |

**Qwen2.5-32B, `budget 8192, 1024 seqs`** (max_num_batched_tokens=8192, max_num_seqs=1024, 15,551 KV blocks)

| trace @ req/s | requests | TTFT p50 (ms) | TTFT p99 (ms) | TPOT p50 (ms) | TPOT p99 (ms) | E2E p99 (s) |
|---|---:|---:|---:|---:|---:|---:|
| code @ 0.5 | 120 | 1114 / 1151 (+3%) | 2829 / 2854 (+1%) | 123.4 / 128.3 (+4%) | 704.4 / 712.0 (+1%) | 10.47 / 10.58 (+1%) |
| code @ 1.0 | 240 | 1149 / 1139 (-1%) | 3092 / 3104 (+0%) | 138.0 / 155.0 (+12%) | 720.0 / 728.7 (+1%) | 10.96 / 11.45 (+4%) |
| code @ 1.5 | 300 | 2249 / 2267 (+1%) | 4447 / 4516 (+2%) | 582.9 / 583.0 (+0%) | 867.6 / 872.4 (+1%) | 48.90 / 49.29 (+1%) |
| code @ 2.0 | 300 | 3066 / 3073 (+0%) | 5833 / 5900 (+1%) | 655.0 / 650.0 (-1%) | 871.5 / 872.1 (+0%) | 48.98 / 49.34 (+1%) |
| conv @ 1.0 | 240 | 132 / 139 (+5%) | 451 / 459 (+2%) | 20.7 / 20.8 (+0%) | 31.9 / 31.5 (-1%) | 11.77 / 11.79 (+0%) |
| conv @ 2.0 | 300 | 150 / 155 (+3%) | 673 / 682 (+1%) | 25.0 / 24.9 (-0%) | 46.3 / 47.0 (+2%) | 14.79 / 14.78 (-0%) |
| conv @ 3.0 | 360 | 168 / 169 (+1%) | 758 / 780 (+3%) | 31.1 / 31.1 (-0%) | 56.4 / 55.6 (-1%) | 18.24 / 18.09 (-1%) |
| conv @ 4.0 | 479 | 187 / 188 (+1%) | 839 / 893 (+6%) | 40.2 / 38.9 (-3%) | 87.2 / 84.9 (-3%) | 23.57 / 23.08 (-2%) |
| conv @ 5.0 | 600 | 269 / 258 (-4%) | 1045 / 1070 (+2%) | 56.6 / 53.1 (-6%) | 104.1 / 103.4 (-1%) | 33.07 / 31.68 (-4%) |
| conv @ 6.0 | 720 | 391 / 384 (-2%) | 1562 / 1626 (+4%) | 85.6 / 79.1 (-8%) | 187.1 / 215.9 (+15%) | 55.09 / 50.60 (-8%) |
</details>

### Which modeling choices matter

The same replays, simulated with one ingredient removed:

| model | TTFT p50 | TTFT p99 | TPOT p50 | TPOT p99 |
|---|---:|---:|---:|---:|
| full model | 4.7% | 2.7% | 3.2% | 1.5% |
| no async scheduling | 10.9% | 4.7% | 3.5% | 2.0% |
| no frontend latency | 5.4% | 3.4% | 3.2% | 1.6% |
| roofline step model | 27.8% | 26.4% | 17.8% | 16.5% |

* **Asynchronous scheduling** is what makes TTFT right. vLLM plans step *k+1* while step *k*
  runs, so a request arriving mid-step waits for an extra step. A simulator that plans
  after each step finishes is about one decode step too optimistic, and that more than
  doubles the median TTFT error.
* **A calibrated step model** is not optional. The datasheet roofline (55% of peak FLOP/s,
  75% of peak bandwidth) is in the right regime but 18–28% off. That is enough to misplace
  saturation entirely.
* **Frontend latency** (HTTP, the API server and streaming; 3.7 ms for 7B) moves TTFT by a
  millisecond or two.

### Choosing a token budget

`max_num_batched_tokens` trades TTFT against TPOT: a large budget lets prefills finish
sooner but stalls decodes behind them. Both settings were measured on the same workloads:

Qwen2.5-7B, `budget 2048, 256 seqs` vs `budget 8192, 1024 seqs` on the same workloads: the simulator orders the two configurations correctly on 27/32 (workload, metric) pairs where they differ by more than 3 %.

| workload | metric | measured budget 2048, 256 seqs vs budget 8192, 1024 seqs | predicted |
|---|---|---:|---:|
| code_r1 | ttft p50 | +4% | +4% |
| code_r1 | ttft p99 | +5% | +1% |
| code_r1 | tpot p50 | +3% | +0% |
| code_r1 | tpot p99 | -9% | -12% |
| code_r2 | ttft p50 | -4% | -16% |
| code_r2 | ttft p99 | +5% | -5% |
| code_r2 | tpot p99 | -48% | -50% |
| code_r3 | ttft p99 | +9% | -3% |
| code_r3 | tpot p50 | -51% | -53% |
| code_r3 | tpot p99 | -73% | -75% |
| code_r3 | e2e p99 | -14% | -20% |
| code_r4 | ttft p99 | +9% | -3% |
| code_r4 | tpot p50 | -47% | -50% |
| code_r4 | tpot p99 | -73% | -75% |
| code_r4 | e2e p99 | -15% | -22% |
| code_r6 | ttft p50 | +11% | -1% |
| code_r6 | ttft p99 | +14% | +1% |
| code_r6 | tpot p50 | -73% | -75% |
| code_r6 | tpot p99 | -73% | -75% |
| code_r6 | e2e p99 | -15% | -22% |
| conv_r4 | ttft p99 | +9% | +8% |
| conv_r16 | ttft p50 | +3% | -2% |
| conv_r20 | ttft p50 | +5% | +3% |
| conv_r20 | ttft p99 | +7% | +3% |
| conv_r24 | ttft p50 | +35% | +12% |
| conv_r24 | ttft p99 | +59% | +11% |
| conv_r24 | tpot p99 | -18% | -16% |
| conv_r28 | ttft p50 | +88% | +72% |
| conv_r28 | ttft p99 | +402% | +326% |
| conv_r28 | tpot p50 | -28% | -28% |
| conv_r28 | tpot p99 | -63% | -63% |
| conv_r28 | e2e p99 | -30% | -33% |

The simulator gets the direction right whenever the difference is larger than its error,
and it predicts the size of the TPOT and end-to-end effects closely. In simulation, the
goodput-maximizing budget for the bursty code trace is 2048: 11% more goodput than the
H200 default of 8192. The conversation trace is insensitive between 1024 and 16384.

### Batching policies

![Policies, Qwen2.5-7B](docs/policies_qwen7b.png)

With the H200-calibrated step models (simulated). Goodput is the highest rate at which 90% of
requests meet the SLO; peak throughput has all requests queued at once.

| policy | 7B conv goodput | 7B conv peak | 7B code peak | 32B conv goodput | 32B conv peak | 32B code peak |
|---|---:|---:|---:|---:|---:|---:|
| static (batch 128) | 0.1 req/s | 5.6 req/s | 4.1 req/s | 0.00 req/s | 1.2 req/s | 1.0 req/s |
| orca (reserve 2048 tokens) | 21.3 req/s | 26.7 req/s | 19.3 req/s | 3.45 req/s | 5.2 req/s | 4.4 req/s |
| orca (oracle reservation) | 21.3 req/s | 26.4 req/s | 19.3 req/s | 3.45 req/s | 5.9 req/s | 4.3 req/s |
| prefill-first (vLLM V0) | 20.1 req/s | 26.6 req/s | 19.5 req/s | 3.19 req/s | 5.9 req/s | 4.3 req/s |
| vLLM V1, budget 8192 | 21.3 req/s | 26.5 req/s | 19.7 req/s | 3.44 req/s | 5.6 req/s | 4.3 req/s |

SLOs: 7B conv TTFT ≤ 500 ms and TPOT ≤ 25 ms; 32B conv TTFT ≤ 1 s and TPOT ≤ 50 ms.

* **Continuous batching is worth 4.3–4.8× the throughput** of static batching, and static
  batching cannot meet an interactive TTFT target at any useful load: a request that
  arrives during a batch waits for that batch's longest output.
* **On an H200, KV memory rarely binds.** With 2.0M tokens of cache for the 7B model, Orca's
  up-front reservation admits as many requests as paged allocation, and Orca and vLLM V1
  make identical decisions whenever prompts fit in the token budget. With the 32B model
  (249k tokens) paging buys 9–15% peak throughput over reserving `max_tokens=2048`, but SLO
  goodput is capped by TPOT before the cache fills.
* **Prefill-first trades TPOT for throughput.** Prefill-only iterations stall every decode,
  which costs 6–7% of goodput under a TPOT target, but they build bigger decode batches and
  reach the highest 32B peak.
* The 32B code-trace SLO is infeasible at every rate. That is consistent with the
  measurements: real vLLM's median TTFT was already 1.1 s at 0.5 req/s, because the trace's
  bursts of up to 40 requests a second queue whole prefills.

## What it models

| policy | batching | KV memory | representative of |
|---|---|---|---|
| `static` | request-level: a batch is padded to its longest prompt and runs to its longest output; results return together | reserved per slot for the padded maximum | FasterTransformer, early TGI, `generate()` loops |
| `orca` | iteration-level: requests join and leave every step; whole prompts are batched with decodes | reserved per request for `prompt + max_tokens` | Orca (OSDI '22) |
| `prefill-first` | iteration-level; prefill-only steps take priority over decode steps | paged, on-demand blocks, LIFO recompute preemption, watermark | vLLM V0 |
| `vllm` | one token budget per step, shared by running requests (decodes and prefill chunks, in admission order) and then FCFS admissions; chunked prefill | paged; LIFO recompute preemption that stops the running pass; no admissions in a preempting step; full-prompt fit check on admission | vLLM V1 (0.34), incl. `long_prefill_token_threshold` and unchunked mode |

The engine models the parts of a real server that move latency:

* **Asynchronous scheduling.** Step *k+1* is planned when step *k* starts, using request state
  as if *k* had completed (vLLM's output placeholders). Requests that arrive during step *k*,
  and memory freed by requests that finish in it, first become visible to the plan for
  *k+2*. A request whose last token is still in flight is skipped but keeps its slot and its
  blocks until that step completes.
* **A calibrated step-time model.** Iteration time is a non-negative linear function of the
  batch's shape. It has a convex piecewise-linear term in the token count padded to vLLM's
  CUDA-graph capture sizes (flat while the weight GEMMs are memory-bound, then
  compute-bound), a per-sequence term, a per-prefill-chunk term, a query-key-pair term for
  prefill attention and a KV-read term for decode attention. It is fitted by NNLS on
  relative error.
* **Frontend latency.** A constant for the HTTP/API-server/streaming path is added to TTFT
  and end-to-end latency. An extra launch delay applies to the first step after the engine
  goes idle. Both are estimated from the calibration replay: requests that found a busy
  engine and were admitted without queueing, and requests that found it idle.

## How the validation works

```
synthetic calibration workload ─▶ vLLM 0.34 + probe on H200 ─▶ every iteration: batch shape, GPU time
                                                            └▶ every request: token arrival times
                        fit step model, estimate frontend latency ◀┘

Azure traces, several rates ───▶ vLLM 0.34 on H200 ──▶ measured TTFT / TPOT / end-to-end
        same actual send times ──▶ batchsim (same limits, same KV block count) ──▶ predicted
```

* **The probe** (`bench/batchsim_vllm_probe.py`) is a vLLM general plugin, not a fork. It
  wraps `Worker.execute_model` and `sample_tokens` with CUDA events that are read lazily, so
  it never stalls a step or perturbs asynchronous scheduling. Timestamps are chained
  step-to-step because `cudaEventElapsedTime`'s float32 milliseconds lose resolution
  against a single reference over a long run. Each record holds the batch's token and
  sequence counts, every prefill chunk's length and context, the decodes' total context,
  and GPU start and end.
* **The client** (`bench/replay.py`) replays a workload open-loop. Prompts are exactly
  `prompt_len` random token ids, with `max_tokens = min_tokens = output_len` and
  `ignore_eos`. It streams the response and records every token's arrival time. Request
  bodies are serialized before the clock starts.
* **Workloads.** Calibration is synthetic: isolated prefills of every length, a decode
  ladder, Poisson traffic at four rates with heavy-tailed lengths, and a 300-request burst.
  Validation uses windows of the Azure LLM inference traces (Patel et al., *Splitwise*,
  ISCA '24), time-scaled to each rate, which keeps their burstiness. The conversation
  trace has Poisson-like arrivals; the code trace's arrival gaps have a coefficient of
  variation of 13.
* **No validation data in the model.** Each model's step model and frontend latency come
  from its own calibration replay. The simulator is given the send times the client
  actually achieved, plus the server's limits and logged KV block count. The second
  configuration (budget 2048) was never used for calibration.
* **Scheduler parity** (`bench/scheduler_parity.py`) builds vLLM's real scheduler on a CPU
  node and feeds it and `batchsim` the same randomized streams. The streams cover tight
  caches that force preemption, chunked and unchunked prefill, long-prefill caps, and sync
  and async engine-core loops.

## Usage

```bash
pip install -e .            # numpy is the only runtime dependency; [plot] adds matplotlib
```

Simulate vLLM on an H200 serving the Azure conversation trace at 12 req/s, with the packaged
step model calibrated on real H200 iterations (it carries its frontend latencies too):

```bash
batchsim simulate --workload azure-conv --n 2000 --rate 12 --cost h200-qwen2.5-7b
```

Compare the four policies on one workload under an SLO:

```bash
batchsim compare --workload azure-conv --n 1500 --rate 16 --cost h200-qwen2.5-7b --slo-ttft 500 --slo-tpot 25
```

Find the highest rate that keeps 90% of requests within an SLO, for several token budgets:

```bash
batchsim capacity --workload azure-code --n 1500 --cost h200-qwen2.5-7b --slo-ttft 500 --slo-e2e 1.5 --budgets 1024 2048 4096 8192
```

Without `--cost`, iteration time comes from a datasheet roofline for `--model` on `--gpu`.
Models: `qwen2.5-0.5b/7b/32b`, `llama-3.1-8b/70b`. GPUs: `h200`, `h100`, `a100-80g`, `b200`,
`l40s`. The KV block count is sized from the same pair unless `--num-blocks` is given.
Workloads can also be `synthetic` (log-normal lengths, Poisson or gamma arrivals) or any CSV
with `arrival,prompt_len,output_len`.

From Python:

```python
from batchsim import SLO, SchedulerConfig, azure, load_cost_model, simulate
from batchsim.engine import EngineConfig

cost = load_cost_model("src/batchsim/data/h200-qwen2.5-7b.json")
wl = azure("conv").head(2000).scale_to_rate(12.0)
cfg = SchedulerConfig(max_num_seqs=1024, max_num_batched_tokens=8192, num_blocks=126399)
eng = EngineConfig(frontend_latency=cost.meta["frontend_latency_s"])
res = simulate(wl, cost, "vllm", cfg, eng)
print(res.summary(slo=SLO(ttft=0.5, tpot=0.025)))
```

## Reproducing the measurements

Everything ran on Northeastern's Explorer cluster; adapt the Slurm headers elsewhere.

1. **vLLM.** The stock `vllm/vllm-openai:v0.34.0` image, converted to Apptainer. The nodes'
   driver (570, CUDA 12.8) is older than the image's CUDA 13, so the job puts the image's
   forward-compatibility libraries on `LD_LIBRARY_PATH`.
2. **Workloads** (committed): `python scripts/make_workloads.py --profile qwen7b --out bench/workloads/qwen7b`.
3. **Serve and replay.** `PROFILE=qwen7b CONFIG=default sbatch bench/explorer.sbatch` serves
   the model with the probe on and smoke-tests probe and client. It then replays the
   calibration workload and every validation workload; if the server dies, it restarts it
   and retries that replay. A second configuration:
   `PROFILE=qwen7b CONFIG=b2048 VLLM_ARGS="--max-num-batched-tokens 2048 --max-num-seqs 256" sbatch bench/explorer.sbatch`.
4. **Scheduler parity:** `SCENARIOS=1000 sbatch bench/parity.sbatch` (CPU only, about two minutes).
5. **Analysis** (any machine, no GPU):
   ```bash
   python scripts/validate.py --profile qwen7b --model-name h200-qwen2.5-7b
   python scripts/experiments.py --out results/h200/qwen7b/experiments.json
   python scripts/make_figures.py
   python scripts/report.py
   ```
   `results/` holds compacted run logs (`scripts/compact_results.py`): probe records and the
   per-request first/last token times. CI re-derives every validation number from them
   (`validate.py --check`).

## Layout

```
src/batchsim/
  schedulers.py   the four policies; `vllm` mirrors vLLM 0.34's Scheduler.schedule()
  engine.py       event loop: async scheduling, idle launch, frontend latency, step log
  cost.py         step-time models: calibrated linear (NNLS) and datasheet roofline
  calibrate.py    probe-log parsing, iteration periods, NNLS fit with held-out error
  kvcache.py      block accounting
  workload.py     synthetic generators, Azure traces, CSV exchange with the bench client
  metrics.py      TTFT / TPOT / end-to-end percentiles, SLO attainment
  sweep.py        rate sweeps and goodput search
  validation.py   measured-vs-predicted comparison of real vLLM runs
  hardware.py     model shapes, GPU datasheets, KV-cache sizing
  data/           packaged H200 step models (Qwen2.5-7B, Qwen2.5-32B)
bench/            GPU side: probe plugin, replay client, Slurm jobs, scheduler parity harness
scripts/          workloads, validation, experiments, figures, README tables
results/          compacted H200 run logs, validation and experiment outputs, parity results
```

## Limitations

* **Long-context prefill chunks.** The calibration replay ran with vLLM's H200 default budget
  of 8192 tokens, so few of its prefill chunks attend to many cached tokens. With a 2048
  budget, long prompts are split into chunks that do. Those iterations take about 5% longer
  than predicted, measured with the corrected probe. That is why TTFT is under-predicted by
  6–12% on the saturated budget-2048 code runs and by up to 35% on the saturated
  conversation runs, where queueing amplifies it. Adding a small-budget segment to the
  calibration replay would cover that shape.
* **Frontend latency is modeled as constants.** The API server admits simultaneous requests
  one at a time, and that queueing is not modeled; it matters only for synchronized bursts.
* **Out of scope:** prefix caching, speculative decoding, LoRA, multimodal inputs,
  tensor/pipeline parallelism and disaggregated prefill. Parity covers the FCFS policy
  without these features.
* **One GPU type measured.** Other GPUs fall back to the roofline model, which the ablation
  shows is 18–28% off. Calibrating one takes a single replay with the probe.
* **Probe timing, first version.** All runs except the last used the first probe, which timed
  every iteration against one reference event. Its float32 milliseconds lose resolution late
  in a job (0.25 ms after 35 minutes), which adds noise to the per-iteration errors of later
  replays. The current probe chains short intervals. Client-side latencies were never
  affected.
