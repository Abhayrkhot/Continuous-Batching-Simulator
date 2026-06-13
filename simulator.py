#!/usr/bin/env python3
"""
Continuous and static batching simulator for LLM inference workloads.

This script generates synthetic sequences with variable output lengths and
simulates two batching strategies:

1. **Static batching** – All sequences in a batch start and finish together.  A
   batch completes when the longest sequence finishes, and new sequences cannot
   be added until then【555339060125421†L166-L190】.

2. **Continuous batching** – Also called iteration‑level scheduling.  At each
   decoding iteration, the server generates one token for every active sequence.
   When a sequence finishes, a new sequence is immediately inserted into the
   decode pool【555339060125421†L212-L229】.

The simulator ignores prompt ingestion (prefill) time for simplicity.  It
computes per‑request end‑to‑end latency (in units of decode steps) and
throughput (tokens per unit time) for both strategies.
"""

from __future__ import annotations

import argparse
import csv
import math
import random
from typing import List, Tuple


def generate_sequence_lengths(
    num_sequences: int,
    dist: str = "exponential",
    mean: float = 100.0,
    std_dev: float = 20.0,
    seed: int | None = None,
) -> List[int]:
    """Generate a list of synthetic sequence lengths.

    Parameters
    ----------
    num_sequences : int
        Number of sequences to generate.
    dist : str
        Distribution to sample from: 'exponential', 'normal' or 'uniform'.
    mean : float
        Mean length of sequences.  For uniform distribution, this is the
        midpoint (range is +/- 50 % around the mean).
    std_dev : float
        Standard deviation for the normal distribution.
    seed : int or None
        Random seed for reproducibility.

    Returns
    -------
    List[int]
        A list of positive integer sequence lengths.
    """
    rng = random.Random(seed)
    lengths: List[int] = []
    if dist == "exponential":
        # Exponential distribution with given mean
        for _ in range(num_sequences):
            # Exponential distribution: mean = 1/lambda => lambda = 1/mean
            lam = 1.0 / mean
            length = max(1, int(rng.expovariate(lam)))
            lengths.append(length)
    elif dist == "normal":
        for _ in range(num_sequences):
            length = int(rng.gauss(mean, std_dev))
            lengths.append(max(1, length))
    elif dist == "uniform":
        # Range is [mean - mean/2, mean + mean/2]
        low = max(1, int(mean * 0.5))
        high = int(mean * 1.5)
        for _ in range(num_sequences):
            length = rng.randint(low, high)
            lengths.append(length)
    else:
        raise ValueError(f"Unsupported distribution: {dist}")
    return lengths


def simulate_static(
    lengths: List[int],
    batch_size: int,
    decode_time: float,
) -> Tuple[float, List[float]]:
    """Simulate static batching and return total time and per‑request latencies.

    Parameters
    ----------
    lengths : list of int
        Sequence lengths (number of tokens) for all requests.
    batch_size : int
        Maximum number of sequences per batch.
    decode_time : float
        Time (in arbitrary units) to generate a single token.

    Returns
    -------
    tuple
        (total_time, latencies)
        where `latencies[i]` is the end‑to‑end latency of the ith sequence.
    """
    total_time = 0.0
    latencies: List[float] = [0.0] * len(lengths)
    # Process in batches of size batch_size
    start_index = 0
    while start_index < len(lengths):
        batch = lengths[start_index : start_index + batch_size]
        max_len = max(batch)
        batch_time = max_len * decode_time
        # Assign the same batch completion time to each request in the batch
        for i, seq_len in enumerate(batch):
            latencies[start_index + i] = total_time + batch_time
        total_time += batch_time
        start_index += batch_size
    return total_time, latencies


def simulate_continuous(
    lengths: List[int],
    batch_size: int,
    decode_time: float,
) -> Tuple[float, List[float]]:
    """Simulate continuous batching and return total time and per‑request latencies.

    Parameters
    ----------
    lengths : list of int
        Sequence lengths (number of tokens) for all requests.
    batch_size : int
        Maximum number of sequences in the decode pool.
    decode_time : float
        Time (in arbitrary units) to generate a single token.

    Returns
    -------
    tuple
        (total_time, latencies)
        where `latencies[i]` is the end‑to‑end latency of the ith sequence.
    """
    # Index of next sequence to be inserted into the decode pool
    next_index = 0
    # Active decode pool: list of remaining token counts
    active: List[int] = []
    # Record completion times for each sequence
    latencies: List[float] = [0.0] * len(lengths)
    # Fill the decode pool to capacity
    while len(active) < batch_size and next_index < len(lengths):
        active.append(lengths[next_index])
        next_index += 1
    time_elapsed = 0.0
    # Track which sequence index corresponds to each active entry
    active_indices = list(range(len(active)))
    # While there are active sequences
    while active:
        # One decode iteration: each active sequence generates one token
        time_elapsed += decode_time
        # Subtract one token from each sequence
        for i in range(len(active)):
            active[i] -= 1
        # Check for finished sequences
        finished_positions = [i for i, remaining in enumerate(active) if remaining <= 0]
        # Process finished sequences from last to first to avoid index shift
        for pos in reversed(finished_positions):
            seq_index = active_indices[pos]
            latencies[seq_index] = time_elapsed
            # Remove finished sequence
            del active[pos]
            del active_indices[pos]
            # Insert a new sequence if available
            if next_index < len(lengths):
                active.insert(pos, lengths[next_index])
                active_indices.insert(pos, next_index)
                next_index += 1
    return time_elapsed, latencies


def compute_throughput(total_tokens: int, total_time: float) -> float:
    """Compute throughput (tokens per unit time)."""
    return total_tokens / total_time if total_time > 0 else 0.0


def run_simulation(args: argparse.Namespace) -> None:
    # Generate synthetic sequence lengths
    lengths = generate_sequence_lengths(
        num_sequences=args.num_sequences,
        dist=args.dist,
        mean=args.mean_length,
        std_dev=args.std_dev,
        seed=args.seed,
    )
    # Simulate static batching
    static_time, static_latencies = simulate_static(lengths, args.batch_size, args.decode_time)
    # Simulate continuous batching
    continuous_time, cont_latencies = simulate_continuous(lengths, args.batch_size, args.decode_time)
    # Compute total tokens (sum of lengths)
    total_tokens = sum(lengths)
    static_tp = compute_throughput(total_tokens, static_time)
    cont_tp = compute_throughput(total_tokens, continuous_time)
    # Compute average latency
    avg_static_lat = sum(static_latencies) / len(static_latencies)
    avg_cont_lat = sum(cont_latencies) / len(cont_latencies)
    # Write results to CSV
    csv_filename = args.output
    with open(csv_filename, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["strategy", "sequence_index", "length", "latency"])
        for i, l in enumerate(lengths):
            writer.writerow(["static", i, l, static_latencies[i]])
        for i, l in enumerate(lengths):
            writer.writerow(["continuous", i, l, cont_latencies[i]])
    # Print summary
    print("Simulation results:")
    print(f"  Static batching:  total time = {static_time:.2f} units, total tokens = {total_tokens}, throughput = {static_tp:.2f} tokens/unit")
    print(f"                    avg latency = {avg_static_lat:.2f} units")
    print(f"  Continuous batching: total time = {continuous_time:.2f} units, total tokens = {total_tokens}, throughput = {cont_tp:.2f} tokens/unit")
    print(f"                      avg latency = {avg_cont_lat:.2f} units")
    print(f"Detailed per‑sequence latencies saved to {csv_filename}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Simulate static vs. continuous batching for LLM workloads.")
    parser.add_argument("--num-sequences", type=int, default=50, help="Number of sequences to simulate")
    parser.add_argument("--batch-size", type=int, default=4, help="Maximum number of active sequences (batch size)")
    parser.add_argument("--dist", type=str, default="exponential", choices=["exponential", "normal", "uniform"], help="Distribution for sequence lengths")
    parser.add_argument("--mean-length", type=float, default=50.0, help="Mean length of sequences (for exponential and normal distributions)")
    parser.add_argument("--std-dev", type=float, default=10.0, help="Standard deviation for normal distribution")
    parser.add_argument("--decode-time", type=float, default=1.0, help="Time per token (arbitrary units)")
    parser.add_argument("--seed", type=int, default=None, help="Random seed")
    parser.add_argument("--output", type=str, default="simulation_results.csv", help="Output CSV filename")
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    run_simulation(args)
