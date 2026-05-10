import asyncio
import logging
import socket
from datetime import datetime, timezone

import httpx
import typer
import uvicorn
from fastapi import FastAPI

from agent.detect import detect_hardware
from common.schemas import Heartbeat

logger = logging.getLogger(__name__)


def _get_local_ip() -> str:
    """Get the LAN IP address of this machine."""
    try:
        # Connect to a public address to determine which interface is used
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


async def heartbeat_loop(
    registry_url: str, machine_id: str, port: int, interval: float = 5.0
):
    specs = detect_hardware()
    logger.info("Detected hardware: %s", specs.model_dump())

    async with httpx.AsyncClient(timeout=5.0) as client:
        while True:
            # Re-detect free memory each heartbeat
            from agent.detect import detect_memory_free_gb
            specs.memory_free_gb = round(detect_memory_free_gb(), 1)

            agent_address = f"http://{_get_local_ip()}:{port}"

            heartbeat = Heartbeat(
                machine_id=machine_id,
                timestamp=datetime.now(timezone.utc),
                specs=specs,
                agent_address=agent_address,
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
