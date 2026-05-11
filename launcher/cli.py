"""Interactive CLI for launching distributed inference."""

import asyncio
import logging
import argparse
import sys

import httpx

logger = logging.getLogger(__name__)


def _format_plan(data: dict) -> str:
    """Format a plan response into a readable summary."""
    plan = data["plan"]
    comparison = data["single_machine_comparison"]
    lines = []

    lines.append("=" * 60)
    lines.append("  DISTRIBUTED INFERENCE PLAN")
    lines.append("=" * 60)
    lines.append("")

    # Assignments
    lines.append("  Machine Assignments:")
    lines.append("  " + "-" * 56)
    for a in plan["assignments"]:
        layers_start, layers_end = a["layers"]
        num_layers = layers_end - layers_start + 1
        lines.append(
            f"    {a['machine_id']:<25} "
            f"layers {layers_start:>3}-{layers_end:<3}  "
            f"({num_layers} layers, {a['memory_required_gb']:.1f}/{a['memory_available_gb']:.1f} GB)"
        )
    lines.append("")

    # Performance
    lines.append("  Performance Estimate:")
    lines.append("  " + "-" * 56)
    lines.append(f"    Predicted throughput:  {plan['predicted_tok_per_sec']:.1f} tok/s")
    lines.append(f"    Time per token:       {plan['total_time_per_token_ms']:.1f} ms")
    lines.append(f"    Compute time:         {plan['compute_time_ms']:.1f} ms")
    lines.append(f"    Network time:         {plan['network_time_ms']:.1f} ms")
    lines.append(f"    Bottleneck:           {plan['bottleneck']}")
    lines.append("")

    # Network transfers
    if plan["network_transfers"]:
        lines.append("  Network Transfers:")
        lines.append("  " + "-" * 56)
        for t in plan["network_transfers"]:
            lines.append(
                f"    {t['from_machine']} -> {t['to_machine']}  "
                f"({t['activation_size_kb']:.1f} KB, {t['latency_ms']:.1f} ms latency)"
            )
        lines.append("")

    # Single machine comparison
    lines.append("  Single Machine Comparison:")
    lines.append("  " + "-" * 56)
    lines.append(f"    Best single machine:  {comparison['best_single_machine']}")
    lines.append(f"    Single-machine tok/s: {comparison['predicted_tok_per_sec']:.1f}")
    lines.append(f"    Verdict:              {comparison['verdict']}")
    lines.append("")
    lines.append("=" * 60)

    return "\n".join(lines)


async def _launch_loop(
    registry_url: str,
    model_name: str | None,
    model_path: str,
    rpc_port: int,
    server_host: str,
    server_port: int,
):
    async with httpx.AsyncClient(timeout=30.0) as client:
        while True:
            # 1. Get plan from registry
            try:
                resp = await client.post(
                    f"{registry_url}/plan",
                    json={"model_name": model_name} if model_name else {},
                )
                resp.raise_for_status()
            except httpx.HTTPStatusError as e:
                detail = e.response.json().get("detail", str(e))
                print(f"\nError fetching plan: {detail}")
                sys.exit(1)
            except httpx.HTTPError as e:
                print(f"\nError fetching plan: {e}")
                sys.exit(1)

            plan_data = resp.json()

            # 2. Show the plan
            print()
            print(_format_plan(plan_data))
            print()

            # 3. Ask for confirmation
            confirm = input("Accept this plan? [y/n]: ").strip().lower()
            if confirm in ("y", "yes"):
                break
            print("\nRe-computing plan...\n")

        # 4. User accepted — call /launch to start RPC workers and get the command
        print("\nStarting RPC workers and preparing launch command...")

        launch_payload = {
            "model_path": model_path,
            "rpc_port": rpc_port,
            "server_host": server_host,
            "server_port": server_port,
        }
        if model_name:
            launch_payload["model_name"] = model_name

        try:
            resp = await client.post(
                f"{registry_url}/launch", json=launch_payload
            )
            resp.raise_for_status()
        except httpx.HTTPError as e:
            print(f"\nLaunch failed: {e}")
            sys.exit(1)

        result = resp.json()
        cmd = result["llamacpp_command"]

        print()
        print("=" * 60)
        print("  RPC workers are running. Run this on the main node:")
        print("=" * 60)
        print()
        print(f"  {cmd['command']}")
        print()
        print(f"  Main machine: {cmd['main_machine']}")
        print(f"  Then prompt at: http://localhost:{server_port}/v1/chat/completions")
        print()


def main():
    parser = argparse.ArgumentParser(description="Launch distributed inference")
    parser.add_argument("--registry", required=True, help="Registry server URL")
    parser.add_argument("--model-name", default=None, help="Preset model name (e.g. mistral-7b-q4)")
    parser.add_argument("--model-path", required=True, help="Path to .gguf model file on main node")
    parser.add_argument("--rpc-port", type=int, default=50052, help="RPC server port on workers")
    parser.add_argument("--server-host", default="0.0.0.0", help="llama-server bind host")
    parser.add_argument("--server-port", type=int, default=8080, help="llama-server port")
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO)
    asyncio.run(
        _launch_loop(args.registry, args.model_name, args.model_path, args.rpc_port, args.server_host, args.server_port)
    )


if __name__ == "__main__":
    main()
