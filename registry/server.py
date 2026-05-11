import asyncio
import logging
import statistics
import time
from contextlib import asynccontextmanager
from datetime import datetime, timezone

import httpx
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel

from common.schemas import Heartbeat
from launcher.command import LlamaCppCommand
from launcher.orchestrate import launch
from planner.engine import plan as compute_plan
from planner.presets import ModelConfig, get_model_config, PRESETS
from registry.models import MachineRecord
from registry.pruner import prune_loop

logger = logging.getLogger(__name__)

machines: dict[str, MachineRecord] = {}


async def probe_latency(agent_address: str, pings: int = 5) -> float | None:
    """Send pings to the agent's /ping endpoint, return median RTT in ms."""
    rtts = []
    async with httpx.AsyncClient(timeout=3.0) as client:
        for _ in range(pings):
            try:
                start = time.monotonic()
                resp = await client.get(f"{agent_address}/ping")
                rtt = (time.monotonic() - start) * 1000
                if resp.status_code == 200:
                    rtts.append(rtt)
            except httpx.HTTPError:
                pass

    if not rtts:
        logger.warning("Latency probe failed for %s", agent_address)
        return None

    median = statistics.median(rtts)
    logger.info("Latency to %s: %.1f ms (median of %d pings)", agent_address, median, len(rtts))
    return round(median, 2)


@asynccontextmanager
async def lifespan(app: FastAPI):
    task = asyncio.create_task(prune_loop(machines))
    yield
    task.cancel()


app = FastAPI(lifespan=lifespan)


@app.post("/heartbeat")
async def heartbeat(hb: Heartbeat):
    is_new = hb.machine_id not in machines

    machines[hb.machine_id] = MachineRecord(
        machine_id=hb.machine_id,
        specs=hb.specs,
        agent_address=hb.agent_address,
        last_seen=datetime.now(timezone.utc),
        latency_ms=machines[hb.machine_id].latency_ms if not is_new else None,
        rpc=hb.rpc,
    )

    if is_new:
        logger.info("New machine registered: %s at %s", hb.machine_id, hb.agent_address)
        asyncio.create_task(_probe_and_update(hb.machine_id, hb.agent_address))

    return {"status": "ok"}


async def _probe_and_update(machine_id: str, agent_address: str):
    latency = await probe_latency(agent_address)
    if machine_id in machines:
        machines[machine_id].latency_ms = latency


@app.get("/machines")
async def get_machines():
    return {
        "machines": [record.model_dump(mode="json") for record in machines.values()]
    }


class PlanRequest(BaseModel):
    model_name: str | None = None
    # Or provide a custom config directly:
    num_layers: int | None = None
    hidden_dim: int | None = None
    precision_bytes: int | None = None
    total_size_gb: float | None = None


def _resolve_model(req: PlanRequest) -> ModelConfig:
    if req.model_name and req.model_name in PRESETS:
        return get_model_config(req.model_name)
    elif req.num_layers and req.hidden_dim and req.precision_bytes and req.total_size_gb:
        return ModelConfig(
            model_name=req.model_name or "custom",
            num_layers=req.num_layers,
            hidden_dim=req.hidden_dim,
            precision_bytes=req.precision_bytes,
            total_size_gb=req.total_size_gb,
        )
    else:
        raise HTTPException(
            status_code=400,
            detail=f"Provide a known model_name ({list(PRESETS.keys())}) or all custom fields",
        )


@app.post("/plan")
async def plan_endpoint(req: PlanRequest):
    if not machines:
        raise HTTPException(status_code=400, detail="No machines online")

    model = _resolve_model(req)

    try:
        result = compute_plan(list(machines.values()), model)
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))

    return result.model_dump(mode="json")


class LaunchRequest(BaseModel):
    model_name: str | None = None
    num_layers: int | None = None
    hidden_dim: int | None = None
    precision_bytes: int | None = None
    total_size_gb: float | None = None
    model_path: str  # path to .gguf file on the main node
    rpc_port: int = 50052
    server_host: str = "0.0.0.0"
    server_port: int = 8080


@app.post("/launch")
async def launch_endpoint(req: LaunchRequest):
    """
    Full orchestration:
    1. Compute optimal plan
    2. Start rpc-server on all worker agents
    3. Wait for readiness
    4. Return the llama-server command
    """
    if not machines:
        raise HTTPException(status_code=400, detail="No machines online")

    plan_req = PlanRequest(
        model_name=req.model_name,
        num_layers=req.num_layers,
        hidden_dim=req.hidden_dim,
        precision_bytes=req.precision_bytes,
        total_size_gb=req.total_size_gb,
    )
    model = _resolve_model(plan_req)

    try:
        plan_result = compute_plan(list(machines.values()), model)
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))

    try:
        command = await launch(
            plan_result=plan_result,
            machines=machines,
            model_path=req.model_path,
            rpc_port=req.rpc_port,
            server_host=req.server_host,
            server_port=req.server_port,
        )
    except RuntimeError as e:
        raise HTTPException(status_code=500, detail=str(e))

    return {
        "plan": plan_result.model_dump(mode="json"),
        "llamacpp_command": command.model_dump(),
    }


@app.get("/health")
async def health():
    return {"status": "healthy", "machines_online": len(machines)}
