from datetime import datetime

from pydantic import BaseModel

from common.schemas import MacSpecs


class MachineRecord(BaseModel):
    machine_id: str
    specs: MacSpecs
    agent_address: str
    last_seen: datetime
    latency_ms: float | None = None
