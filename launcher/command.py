"""Generate llama.cpp launch commands from a plan."""

from pydantic import BaseModel

from planner.scorer import PlanScore


class LlamaCppCommand(BaseModel):
    main_machine: str
    rpc_endpoints: str  # comma-separated host:port
    tensor_split: str  # comma-separated ratios
    command: str


def build_command(
    plan: PlanScore,
    model_path: str,
    machine_addresses: dict[str, str],  # machine_id -> LAN IP
    rpc_ports: dict[str, int],  # machine_id -> rpc port
    host: str = "0.0.0.0",
    port: int = 8080,
) -> LlamaCppCommand:
    """
    Build a llama-server command from a plan.

    The first machine in the plan is the main node (runs llama-server).
    The rest are RPC workers.
    """
    assignments = plan.assignments
    total_layers = assignments[-1].layers[1] + 1

    # Main machine is first in the pipeline
    main_id = assignments[0].machine_id

    # Tensor split ratios — layers per machine / total layers
    ratios = []
    for a in assignments:
        num_layers = a.layers[1] - a.layers[0] + 1
        ratios.append(num_layers / total_layers)
    tensor_split = ",".join(f"{r:.2f}" for r in ratios)

    # RPC endpoints — all machines except the main node
    rpc_parts = []
    for a in assignments[1:]:
        ip = machine_addresses[a.machine_id]
        rpc_port = rpc_ports.get(a.machine_id, 50052)
        rpc_parts.append(f"{ip}:{rpc_port}")
    rpc_endpoints = ",".join(rpc_parts)

    # Build command
    cmd_parts = [
        "llama-server",
        f"-m {model_path}",
        "-ngl 99",
    ]
    if rpc_endpoints:
        cmd_parts.append(f"--rpc {rpc_endpoints}")
    cmd_parts.append(f"--tensor-split {tensor_split}")
    cmd_parts.append(f"--host {host}")
    cmd_parts.append(f"--port {port}")

    return LlamaCppCommand(
        main_machine=main_id,
        rpc_endpoints=rpc_endpoints,
        tensor_split=tensor_split,
        command=" ".join(cmd_parts),
    )
