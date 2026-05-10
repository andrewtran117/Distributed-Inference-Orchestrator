from pydantic import BaseModel
from datetime import datetime


class MacSpecs(BaseModel):
    chip: str
    memory_total_gb: float
    memory_free_gb: float
    memory_bandwidth_gbs: float
    gpu_cores: int


class Heartbeat(BaseModel):
    machine_id: str
    timestamp: datetime
    specs: MacSpecs
