"""ARQ worker configuration."""

from __future__ import annotations

from arq.connections import RedisSettings

from app.config import get_settings
from app.jobs.tasks import ingest_task


class WorkerSettings:
    functions = [ingest_task]
    redis_settings = RedisSettings.from_dsn(get_settings().REDIS_URL)
