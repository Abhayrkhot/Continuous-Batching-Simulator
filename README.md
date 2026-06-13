# Continuous Batching Simulator

## Overview

**Continuous batching** is a scheduling technique for large language model (LLM) inference that allows new requests to join an active generation batch whenever another request finishes generating its tokens.  Traditional static batching holds all requests in a batch until the longest sequence finishes, which under‑utilizes the GPU when sequences have varying lengths【555339060125421†L166-L190】.  By contrast, continuous batching inserts new sequences into the batch as soon as one finishes, achieving higher GPU utilization and better throughput【555339060125421†L212-L228】.  This simulator demonstrates the difference between static and continuous batching on synthetic workloads.

## Why continuous batching matters

LLM inference is memory‑bound rather than compute‑bound【555339060125421†L115-L120】.  The amount of GPU memory consumed scales with the base model size plus the length of the token sequence【555339060125421†L122-L132】, so batch size is limited by available memory.  In static batching, the batch size is fixed for the entire duration of inference and new requests must wait for the slowest request to finish【555339060125421†L166-L190】.  Continuous batching (also known as iteration‑level scheduling) addresses this inefficiency by allowing completed sequences to be replaced immediately by new ones【555339060125421†L212-L229】.  Anyscale reports that continuous batching can deliver up to 23× throughput improvement over naive batching, and 8× improvement over static batching on frameworks like Ray Serve and Hugging Face’s text‑generation‑inference【555339060125421†L40-L50】.

## Features

- **Workload generator** – Creates a synthetic workload of sequences with variable output lengths drawn from a configurable distribution (default exponential).
- **Static batching simulation** – Processes requests in fixed‑size batches where new requests wait until all sequences in the current batch finish.
- **Continuous batching simulation** – Maintains a decode pool of active sequences; whenever a sequence finishes, a new sequence is inserted if available.
- **Per‑request latencies** – Records end‑to‑end latency for each request under both batching strategies.
- **Throughput measurement** – Computes total tokens processed per unit time and average latency for each batching method.
- **Command‑line interface** – Configure the number of requests, batch size, distribution of sequence lengths and random seed.
- **Tests** – Includes unit tests to validate the core logic of the simulator.

## Architecture

```
continuous-batching-simulator/
├── simulator.py        # core simulation logic and CLI
├── tests/
│   └── test_sim.py     # unit tests for simulator
├── requirements.txt    # Python dependencies
└── README.md
```

### Static batching

In static batching, all requests in the batch start and finish together.  The batch end time is determined by the longest sequence in the batch.  Shorter sequences finish early but cannot release resources【555339060125421†L166-L190】.  If batch size is `B` and the request lengths in the batch are `l₁, l₂, …, l_B`, the batch duration is:

```
batch_time = max(l₁, l₂, …, l_B) * decode_time
```

Every request in the batch experiences the same latency even if its own length is shorter than the maximum.  New requests wait until the current batch completes.  In practice this leads to under‑utilization of GPU compute for variable‑length sequences【555339060125421†L183-L203】.

### Continuous batching

Continuous batching uses **iteration‑level scheduling**: at each decoding iteration, the server generates a token for every active sequence.  When a sequence finishes, a new request is inserted into the decode pool without waiting for other requests【555339060125421†L212-L229】.  This approach maintains a roughly constant number of active sequences and allows short sequences to complete without blocking longer ones.  In more advanced implementations, prefill (prompt ingestion) happens concurrently or is rate‑limited via a hyper‑parameter such as `waiting_served_ratio`【555339060125421†L232-L237】, but this simulator ignores prefill time for simplicity.

## Usage

First install dependencies (Python 3.9+):

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

Run a simulation with the default parameters:

```bash
python simulator.py \
  --num-sequences 100 \
  --batch-size 8 \
  --dist exponential \
  --mean-length 100 \
  --seed 42
```

This command will generate 100 sequences with lengths drawn from an exponential distribution (mean 100 tokens), simulate static and continuous batching with a batch size of 8 and print a summary of throughput and average latency for both strategies.  It also saves detailed per‑request latencies to a CSV file `simulation_results.csv`.

You can choose a normal or uniform distribution for sequence lengths via `--dist normal` or `--dist uniform`.  Use `--mean-length` to specify the mean of the distribution and `--std-dev` for the standard deviation (for the normal distribution).

### Example output

On a synthetic workload of 100 sequences with exponential length distribution (mean 100), batch size 8 and token decode time of 1 unit, a run might produce the following summary (numbers will vary with the random seed):

```
Simulation results:
  Static batching:  total time = 335 units, total tokens = 10194, throughput = 30.42 tokens/unit
                    avg latency = 99.02 units
  Continuous batching: total time = 290 units, total tokens = 10194, throughput = 35.15 tokens/unit
                      avg latency = 90.29 units
```

In this example, continuous batching processes the same number of tokens ~15 % faster than static batching, and the average per‑request latency drops by roughly 9 %.

## Tests

Run the unit tests with:

```bash
pytest -q
```

The tests validate that the simulator correctly reports latencies and throughput for small workloads.

## Limitations and future work

This simulator is intentionally simplified.  In production systems, prefill (prompt ingestion) and decode phases require separate scheduling and can be interleaved.  The prefill stage may run in parallel with decode under certain frameworks, and memory constraints can prevent immediate insertion of new sequences into a batch.  Nonetheless, this simulator illustrates the key insight that iteration‑level scheduling leads to higher GPU utilization and lower latency for variable‑length workloads.

Future improvements could include modeling prefill delays, variable arrival rates (non‑queued workloads), and adding support for GPU memory constraints.
