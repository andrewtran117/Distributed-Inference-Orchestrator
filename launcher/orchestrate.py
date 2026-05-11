"""Orchestrate rpc-server startup across agents and launch inference."""

import asyncio
import logging

import httpx

from launcher.command import LlamaCppCommand, build_command
from planner.engine import PlanResult
from registry.models import MachineRecord

logger = logging.getLogger(__name__)

READY_POLL_INTERVAL = 0.5
READY_TIMEOUT = 15.0


async def start_rpc_on_agent(
    client: httpx.AsyncClient, agent_address: str, rpc_port: int
) -> bool:
    """Tell an agent to start its rpc-server. Returns True on success."""
    try:
        resp = await client.post(
            f"{agent_address}/rpc/start", json={"port": rpc_port}
        )
        resp.raise_for_status()
        return True
    except httpx.HTTPError as e:
        logger.error("Failed to start rpc-server on %s: %s", agent_address, e)
        return False


async def wait_for_rpc_ready(
    client: httpx.AsyncClient, agent_address: str
) -> bool:
    """Poll agent until rpc-server is running or timeout."""
    elapsed = 0.0
    while elapsed < READY_TIMEOUT:
        try:
            resp = await client.get(f"{agent_address}/rpc/status")
            if resp.status_code == 200 and resp.json().get("running"):
                return True
        except httpx.HTTPError:
            pass
        await asyncio.sleep(READY_POLL_INTERVAL)
        elapsed += READY_POLL_INTERVAL
    return False


async def launch(
    plan_result: PlanResult,
    machines: dict[str, MachineRecord],
    model_path: str,
    rpc_port: int = 50052,
    server_host: str = "0.0.0.0",
    server_port: int = 8080,
) -> LlamaCppCommand:
    """
    Full launch orchestration:
    1. Start rpc-server on all worker agents
    2. Wait for them to be ready
    3. Return the llama-server command for the main node
    """
    plan = plan_result.plan
    assignments = plan.assignments
    main_id = assignments[0].machine_id

    # Worker machines = everyone except the main node
    worker_ids = [a.machine_id for a in assignments[1:]]

    async with httpx.AsyncClient(timeout=10.0) as client:
        # 1. Start rpc-server on all workers in parallel
        start_tasks = []
        for wid in worker_ids:
            record = machines[wid]
            start_tasks.append(
                start_rpc_on_agent(client, record.agent_address, rpc_port)
            )

        results = await asyncio.gather(*start_tasks)
        failed = [
            wid for wid, ok in zip(worker_ids, results) if not ok
        ]
        if failed:
            raise RuntimeError(
                f"Failed to start rpc-server on: {', '.join(failed)}"
            )

        # 2. Wait for all workers to be ready
        ready_tasks = []
        for wid in worker_ids:
            record = machines[wid]
            ready_tasks.append(wait_for_rpc_ready(client, record.agent_address))

        ready_results = await asyncio.gather(*ready_tasks)
        not_ready = [
            wid for wid, ok in zip(worker_ids, ready_results) if not ok
        ]
        if not_ready:
            raise RuntimeError(
                f"rpc-server not ready on: {', '.join(not_ready)}"
            )

    # 3. Build the llama-server command
    machine_addresses = {}
    rpc_ports = {}
    for a in assignments:
        record = machines[a.machine_id]
        # Extract IP from agent_address (e.g. "http://192.168.1.10:9001" -> "192.168.1.10")
        ip = record.agent_address.split("//")[1].split(":")[0]
        machine_addresses[a.machine_id] = ip
        rpc_ports[a.machine_id] = rpc_port

    command = build_command(
        plan=plan,
        model_path=model_path,
        machine_addresses=machine_addresses,
        rpc_ports=rpc_ports,
        host=server_host,
        port=server_port,
    )

    logger.info("Launch ready. Main node: %s", main_id)
    logger.info("Command: %s", command.command)

    return command
