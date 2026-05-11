import asyncio
import logging
import shutil
import signal
import socket
from datetime import datetime, timezone

import httpx
import typer
import uvicorn
from fastapi import FastAPI
from pydantic import BaseModel

from agent.detect import detect_hardware
from common.schemas import Heartbeat, RpcStatus

logger = logging.getLogger(__name__)

rpc_process: asyncio.subprocess.Process | None = None
rpc_status = RpcStatus()


def _get_local_ip() -> str:
    """Get the LAN IP address of this machine."""
    try:
        s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        s.connect(("8.8.8.8", 80))
        ip = s.getsockname()[0]
        s.close()
        return ip
    except OSError:
        return "127.0.0.1"


ping_app = FastAPI()


@ping_app.get("/ping")
async def ping():
    return {"status": "ok"}


# --- RPC management endpoints (Phase B) ---


class RpcStartRequest(BaseModel):
    port: int = 50052


@ping_app.post("/rpc/start")
async def rpc_start(req: RpcStartRequest):
    global rpc_process, rpc_status

    if rpc_process and rpc_process.returncode is None:
        return {"status": "already_running", "port": rpc_status.port}

    rpc_server_bin = shutil.which("rpc-server")
    if not rpc_server_bin:
        return {"status": "error", "detail": "rpc-server binary not found in PATH"}

    rpc_process = await asyncio.create_subprocess_exec(
        rpc_server_bin,
        "--host", "0.0.0.0",
        "--port", str(req.port),
        stdout=asyncio.subprocess.PIPE,
        stderr=asyncio.subprocess.PIPE,
    )
    rpc_status = RpcStatus(running=True, port=req.port)
    logger.info("Started rpc-server on port %d (pid %d)", req.port, rpc_process.pid)
    return {"status": "started", "port": req.port, "pid": rpc_process.pid}


@ping_app.post("/rpc/stop")
async def rpc_stop():
    global rpc_process, rpc_status

    if not rpc_process or rpc_process.returncode is not None:
        rpc_status = RpcStatus()
        return {"status": "not_running"}

    rpc_process.send_signal(signal.SIGTERM)
    try:
        await asyncio.wait_for(rpc_process.wait(), timeout=5.0)
    except asyncio.TimeoutError:
        rpc_process.kill()

    rpc_status = RpcStatus()
    rpc_process = None
    logger.info("Stopped rpc-server")
    return {"status": "stopped"}


@ping_app.get("/rpc/status")
async def rpc_status_endpoint():
    global rpc_process, rpc_status

    # Check if the process died unexpectedly
    if rpc_process and rpc_process.returncode is not None:
        rpc_status = RpcStatus()
        rpc_process = None

    return rpc_status.model_dump()


# --- Heartbeat loop ---


async def heartbeat_loop(
    registry_url: str, machine_id: str, port: int, interval: float = 5.0
):
    specs = detect_hardware()
    logger.info("Detected hardware: %s", specs.model_dump())

    async with httpx.AsyncClient(timeout=5.0) as client:
        while True:
            from agent.detect import detect_memory_free_gb
            specs.memory_free_gb = round(detect_memory_free_gb(), 1)

            agent_address = f"http://{_get_local_ip()}:{port}"

            heartbeat = Heartbeat(
                machine_id=machine_id,
                timestamp=datetime.now(timezone.utc),
                specs=specs,
                agent_address=agent_address,
                rpc=rpc_status,
            )

            try:
                resp = await client.post(
                    f"{registry_url}/heartbeat",
                    json=heartbeat.model_dump(mode="json"),
                )
                resp.raise_for_status()
                logger.debug("Heartbeat sent successfully")
            except httpx.HTTPError as e:
                logger.warning("Heartbeat failed: %s", e)

            await asyncio.sleep(interval)


async def run(registry_url: str, port: int, machine_id: str):
    config = uvicorn.Config(ping_app, host="0.0.0.0", port=port, log_level="info")
    server = uvicorn.Server(config)

    heartbeat_task = asyncio.create_task(
        heartbeat_loop(registry_url, machine_id, config.port)
    )

    try:
        await server.serve()
    finally:
        heartbeat_task.cancel()
        # Clean up rpc-server on shutdown
        if rpc_process and rpc_process.returncode is None:
            rpc_process.send_signal(signal.SIGTERM)


cli = typer.Typer()


@cli.command()
def main(
    registry: str = typer.Option(..., help="Registry server URL"),
    port: int = typer.Option(9001, help="Port for /ping endpoint"),
    machine_id: str = typer.Option(
        None, help="Machine ID (defaults to hostname)"
    ),
):
    """Start the agent: detect hardware, heartbeat to registry, serve /ping."""
    logging.basicConfig(level=logging.INFO)

    if machine_id is None:
        machine_id = f"mac-{socket.gethostname().split('.')[0].lower()}"

    logger.info("Starting agent %s on port %d -> %s", machine_id, port, registry)
    asyncio.run(run(registry, port, machine_id))


if __name__ == "__main__":
    cli()
