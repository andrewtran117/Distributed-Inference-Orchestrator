import asyncio
import logging
from datetime import datetime, timezone

logger = logging.getLogger(__name__)

PRUNE_INTERVAL_S = 10
STALE_THRESHOLD_S = 15


async def prune_loop(machines: dict):
    """Remove machines that haven't sent a heartbeat in STALE_THRESHOLD_S seconds."""
    while True:
        await asyncio.sleep(PRUNE_INTERVAL_S)
        now = datetime.now(timezone.utc)
        stale = [
            mid for mid, record in machines.items()
            if (now - record.last_seen).total_seconds() > STALE_THRESHOLD_S
        ]
        for mid in stale:
            logger.info("Pruning stale machine: %s", mid)
            del machines[mid]
