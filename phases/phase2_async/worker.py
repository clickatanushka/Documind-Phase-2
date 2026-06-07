"""
phases/phase2_async/worker.py
Celery app configuration.

Start the worker with:
    celery -A phases.phase2_async.worker worker --loglevel=info
"""

import os
from celery import Celery
from dotenv import load_dotenv

load_dotenv()

BROKER  = os.getenv("CELERY_BROKER_URL",    "redis://localhost:6379/0")
BACKEND = os.getenv("CELERY_RESULT_BACKEND","redis://localhost:6379/0")

celery_app = Celery(
    "documind",
    broker=BROKER,
    backend=BACKEND,
    include=["phases.phase2_async.tasks"],
)

celery_app.conf.update(
    task_serializer="json",
    result_serializer="json",
    accept_content=["json"],
    task_track_started=True,       # STARTED state visible via AsyncResult
    result_expires=3600,           # job results kept for 1 hour
    worker_prefetch_multiplier=1,  # one task at a time per worker (PDF-heavy)
)
