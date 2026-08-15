import os
import asyncio
import time
import uuid
import logging
from enum import Enum
from typing import Dict, Optional, List
from contextlib import asynccontextmanager

from fastapi import FastAPI, HTTPException, Request
from fastapi.staticfiles import StaticFiles
from fastapi.responses import JSONResponse, FileResponse

import config
from app import Input, Output, ZImageApp

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(name)s] %(levelname)s: %(message)s",
)
logger = logging.getLogger("zimage.server")


class JobStatus(str, Enum):
    IN_QUEUE = "IN_QUEUE"
    IN_PROGRESS = "IN_PROGRESS"
    COMPLETED = "COMPLETED"
    FAILED = "FAILED"
    CANCELLED = "CANCELLED"


class Job:
    def __init__(self, request_id: str, input_data: Input):
        self.request_id = request_id
        self.input = input_data
        self.status = JobStatus.IN_QUEUE
        self.result: Optional[Output] = None
        self.error: Optional[str] = None
        self.logs: List[dict] = []
        self.created_at = time.time()
        self.started_at: Optional[float] = None
        self.completed_at: Optional[float] = None

    def to_status_dict(self) -> dict:
        d = {
            "status": self.status.value,
            "request_id": self.request_id,
            "logs": self.logs,
        }
        if self.status == JobStatus.IN_QUEUE:
            ahead = sum(
                1
                for j in jobs.values()
                if j.status == JobStatus.IN_QUEUE and j.created_at < self.created_at
            )
            d["queue_position"] = ahead
        if self.status == JobStatus.COMPLETED:
            d["result_url"] = f"/result/{self.request_id}"
        if self.status == JobStatus.FAILED:
            d["error"] = self.error
        if self.started_at and self.completed_at:
            d["processing_seconds"] = round(self.completed_at - self.started_at, 2)
        return d


jobs: Dict[str, Job] = {}
job_queue: asyncio.Queue = None
zimage_app: ZImageApp = None
startup_error: Optional[str] = None
startup_complete: bool = False
_cleanup_counter = 0


async def worker_loop():
    logger.info("Worker started")
    global _cleanup_counter
    while True:
        job = await job_queue.get()

        if job.status == JobStatus.CANCELLED:
            job_queue.task_done()
            continue

        job.status = JobStatus.IN_PROGRESS
        job.started_at = time.time()
        job.logs.append(
            {"message": f"Starting generation: {job.input.prompt[:80]}"}
        )

        retries = 0
        while retries <= config.MAX_RETRIES:
            try:
                loop = asyncio.get_event_loop()
                result = await loop.run_in_executor(
                    None, zimage_app.generate, job.input
                )
                job.result = result
                job.status = JobStatus.COMPLETED
                job.logs.append({"message": "Generation completed"})
                break
            except Exception as e:
                retries += 1
                if retries <= config.MAX_RETRIES:
                    job.logs.append(
                        {"message": f"Attempt {retries} failed: {e}, retrying..."}
                    )
                    logger.warning(
                        "Job %s attempt %d failed: %s", job.request_id, retries, e
                    )
                else:
                    job.error = str(e)
                    job.status = JobStatus.FAILED
                    job.logs.append(
                        {"message": f"All retries exhausted: {e}"}
                    )
                    logger.error(
                        "Job %s failed after %d retries: %s",
                        job.request_id,
                        retries,
                        e,
                    )

        job.completed_at = time.time()
        elapsed = job.completed_at - (job.started_at or job.completed_at)
        logger.info(
            "Job %s: %s in %.1fs", job.request_id, job.status.value, elapsed
        )
        job_queue.task_done()

        _cleanup_counter += 1
        if _cleanup_counter % 10 == 0:
            _cleanup_old_jobs()


def _cleanup_old_jobs():
    now = time.time()
    expired = [
        rid
        for rid, j in jobs.items()
        if j.status in (JobStatus.COMPLETED, JobStatus.FAILED, JobStatus.CANCELLED)
        and j.completed_at
        and now - j.completed_at > config.JOB_TTL_SECONDS
    ]
    for rid in expired:
        del jobs[rid]
    if expired:
        logger.info("Cleaned up %d expired jobs", len(expired))


@asynccontextmanager
async def lifespan(app: FastAPI):
    global job_queue, zimage_app, startup_complete, startup_error
    job_queue = asyncio.Queue(maxsize=config.MAX_QUEUE_SIZE)

    logger.info("Starting Z-Image-Turbo server (load_mode=%s)", config.LOAD_MODE)

    loop = asyncio.get_event_loop()
    try:
        zimage_app = ZImageApp()
        await loop.run_in_executor(None, zimage_app.setup)
        startup_complete = True
        logger.info("Server ready")
    except Exception as e:
        startup_error = str(e)
        logger.error("Startup failed: %s", e, exc_info=True)

    worker = asyncio.create_task(worker_loop())

    yield

    worker.cancel()
    try:
        await worker
    except asyncio.CancelledError:
        pass


app = FastAPI(
    title="Z-Image-Turbo Inference Server",
    description="FAL.ai-style async queue API for Z-Image-Turbo image generation",
    version="1.0.0",
    lifespan=lifespan,
)

os.makedirs(config.OUTPUT_DIR, exist_ok=True)
app.mount("/output", StaticFiles(directory=config.OUTPUT_DIR), name="output")
app.mount("/static", StaticFiles(directory="static"), name="static")


@app.get("/health")
async def health():
    if startup_error:
        return JSONResponse(
            status_code=503,
            content={"status": "error", "error": startup_error},
        )
    if not startup_complete:
        return JSONResponse(
            status_code=503,
            content={"status": "starting", "message": "Model loading..."},
        )
    return {"status": "ok"}


@app.post("/submit")
async def submit(input: Input, request: Request):
    if not startup_complete:
        raise HTTPException(
            status_code=503, detail="Server is still starting up"
        )
    if startup_error:
        raise HTTPException(
            status_code=503, detail=f"Server startup failed: {startup_error}"
        )

    request_id = str(uuid.uuid4())
    job = Job(request_id, input)
    jobs[request_id] = job

    try:
        job_queue.put_nowait(job)
    except asyncio.QueueFull:
        del jobs[request_id]
        raise HTTPException(
            status_code=429, detail="Queue is full, try again later"
        )

    base = str(request.base_url).rstrip("/")
    return {
        "request_id": request_id,
        "status": JobStatus.IN_QUEUE.value,
        "status_url": f"{base}/status/{request_id}",
        "result_url": f"{base}/result/{request_id}",
        "cancel_url": f"{base}/cancel/{request_id}",
    }


@app.post("/generate")
async def generate(input: Input, request: Request):
    if not startup_complete:
        raise HTTPException(
            status_code=503, detail="Server is still starting up"
        )
    if startup_error:
        raise HTTPException(
            status_code=503, detail=f"Server startup failed: {startup_error}"
        )

    loop = asyncio.get_event_loop()
    result = await loop.run_in_executor(
        None, zimage_app.generate_sync, input
    )
    return result.model_dump()


@app.get("/status/{request_id}")
async def get_status(request_id: str):
    job = jobs.get(request_id)
    if job is None:
        raise HTTPException(status_code=404, detail="Request not found")
    return job.to_status_dict()


@app.get("/result/{request_id}")
async def get_result(request_id: str):
    job = jobs.get(request_id)
    if job is None:
        raise HTTPException(status_code=404, detail="Request not found")
    if job.status == JobStatus.COMPLETED and job.result:
        return job.result.model_dump()
    if job.status == JobStatus.FAILED:
        raise HTTPException(
            status_code=422, detail=f"Generation failed: {job.error}"
        )
    return JSONResponse(
        status_code=202,
        content={
            "status": job.status.value,
            "message": "Result not ready yet",
            "status_url": f"/status/{request_id}",
        },
    )


@app.post("/cancel/{request_id}")
async def cancel(request_id: str):
    job = jobs.get(request_id)
    if job is None:
        raise HTTPException(status_code=404, detail="Request not found")
    if job.status in (JobStatus.COMPLETED, JobStatus.FAILED, JobStatus.CANCELLED):
        return {
            "request_id": request_id,
            "status": job.status.value,
            "message": "Already finished",
        }
    if job.status == JobStatus.IN_QUEUE:
        job.status = JobStatus.CANCELLED
        return {
            "request_id": request_id,
            "status": JobStatus.CANCELLED.value,
            "message": "Cancelled",
        }
    return {
        "request_id": request_id,
        "status": job.status.value,
        "message": "Cannot cancel in-progress job",
    }


@app.get("/")
async def root():
    return FileResponse("static/index.html")


@app.get("/api/history")
async def api_history():
    import os as _os
    output_dir = config.OUTPUT_DIR
    files = []
    if _os.path.isdir(output_dir):
        for name in _os.listdir(output_dir):
            if name.startswith("."):
                continue
            filepath = _os.path.join(output_dir, name)
            if not _os.path.isfile(filepath):
                continue
            mtime = _os.path.getmtime(filepath)
            files.append({"url": f"/output/{name}", "name": name, "mtime": mtime})
    files.sort(key=lambda f: f["mtime"], reverse=True)
    files = files[:20]
    return {"images": files}
