"""Score a single candidate layer assignment."""

from pydantic import BaseModel

from planner.presets import ModelConfig
from registry.models import MachineRecord

LINK_SPEED_BYTES_PER_SEC = 125 * 1e6  # 1 Gbps


class Assignment(BaseModel):
    machine_id: str
    layers: tuple[int, int]  # (start, end) inclusive
    memory_required_gb: float
    memory_available_gb: float
    compute_time_ms: float


class NetworkTransfer(BaseModel):
    from_machine: str
    to_machine: str
    activation_size_kb: float
    latency_ms: float
    transfer_time_ms: float


class PlanScore(BaseModel):
    assignments: list[Assignment]
    network_transfers: list[NetworkTransfer]
    predicted_tok_per_sec: float
    bottleneck: str
    total_time_per_token_ms: float
    compute_time_ms: float
    network_time_ms: float


def score_assignment(
    machines: list[MachineRecord],
    layer_counts: list[int],
    model: ModelConfig,
) -> PlanScore | None:
    """
    Score a candidate assignment.

    machines: ordered list of machines in pipeline order.
    layer_counts: how many layers each machine gets (same order).
    model: the model config.

    Returns a PlanScore, or None if the assignment is invalid (memory overflow).
    """
    size_per_layer = model.total_size_gb / model.num_layers
    assignments: list[Assignment] = []
    compute_times: list[float] = []

    layer_start = 0
    for machine, count in zip(machines, layer_counts):
        if count == 0:
            continue

        layer_end = layer_start + count - 1
        weight_size = size_per_layer * count

        if weight_size > machine.specs.memory_free_gb:
            return None  # doesn't fit

        compute_ms = (weight_size / machine.specs.memory_bandwidth_gbs) * 1000
        compute_times.append(compute_ms)

        assignments.append(Assignment(
            machine_id=machine.machine_id,
            layers=(layer_start, layer_end),
            memory_required_gb=round(weight_size, 2),
            memory_available_gb=machine.specs.memory_free_gb,
            compute_time_ms=round(compute_ms, 2),
        ))

        layer_start = layer_end + 1

    # Network transfers between adjacent stages
    activation_bytes = model.hidden_dim * model.precision_bytes
    activation_kb = activation_bytes / 1024
    transfers: list[NetworkTransfer] = []
    network_times: list[float] = []

    for i in range(len(assignments) - 1):
        m_from = assignments[i]
        m_to = assignments[i + 1]

        # Look up latency from the machine records
        from_record = next(m for m in machines if m.machine_id == m_from.machine_id)
        to_record = next(m for m in machines if m.machine_id == m_to.machine_id)

        latency = max(
            from_record.latency_ms or 0.0,
            to_record.latency_ms or 0.0,
        )
        transfer_time_ms = (activation_bytes / LINK_SPEED_BYTES_PER_SEC) * 1000
        hop_time = transfer_time_ms + latency
        network_times.append(hop_time)

        transfers.append(NetworkTransfer(
            from_machine=m_from.machine_id,
            to_machine=m_to.machine_id,
            activation_size_kb=round(activation_kb, 2),
            latency_ms=round(latency, 2),
            transfer_time_ms=round(transfer_time_ms, 4),
        ))

    total_compute = sum(compute_times)
    total_network = sum(network_times)
    total_time = total_compute + total_network

    return PlanScore(
        assignments=assignments,
        network_transfers=transfers,
        predicted_tok_per_sec=round(1000 / total_time, 1),
        bottleneck="network" if total_network > total_compute else "compute",
        total_time_per_token_ms=round(total_time, 2),
        compute_time_ms=round(total_compute, 2),
        network_time_ms=round(total_network, 2),
    )
