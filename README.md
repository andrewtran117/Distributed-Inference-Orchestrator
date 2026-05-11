# Distributed-Inference-Orchestrator

A lightweight system that discovers Apple Silicon Macs on a network, profiles their hardware (memory, bandwidth, GPU cores), and computes the optimal layer split for running LLM inference across multiple machines. It generates ready-to-run llama.cpp commands with the correct `--rpc` endpoints and `--tensor-split` ratios, so you don't have to figure out the distribution yourself. Agents heartbeat their specs to a central registry, and an interactive CLI lets you review the plan before launching.
