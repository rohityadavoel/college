# ============================================================
#  main.py  ·  Skin Disease Detection API  ·  FastAPI Edition
#  Author   : You
#  Speed    : Async I/O · Thread/Process pools · RAM cache
# ============================================================

from __future__ import annotations

import asyncio
import json
import multiprocessing
import os
import traceback
import uuid
import warnings
from concurrent.futures import ProcessPoolExecutor, ThreadPoolExecutor
from contextlib import asynccontextmanager
from datetime import datetime
from functools import lru_cache
from pathlib import Path
from typing import Any, Optional

import cv2
import numpy as np
import sqlite3
import uvicorn
from fastapi import BackgroundTasks, FastAPI, File, HTTPException, Request, UploadFile, status
from fastapi.responses import HTMLResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
from pydantic import BaseModel, Field
from rich.console import Console
from rich.logging import RichHandler
from rich.panel import Panel
from rich.table import Table
from rich import box
import logging
import time

import config
from utils.model_loader import load_keras_model, predict_binary, predict_stage2
from validators.fast_validator import validate_image

# ─────────────────────────────────────────────
#  GLOBAL SETUP
# ─────────────────────────────────────────────
warnings.filterwarnings("ignore")
os.environ["CUDA_VISIBLE_DEVICES"] = "-1"

# Rich console for pretty terminal output
console = Console()

# Logging via Rich
logging.basicConfig(
    level=logging.INFO,
    format="%(message)s",
    datefmt="[%X]",
    handlers=[RichHandler(console=console, rich_tracebacks=True, markup=True)],
)
log = logging.getLogger("skin-api")


# ─────────────────────────────────────────────
#  PYDANTIC RESPONSE MODELS
# ─────────────────────────────────────────────
class BinaryPrediction(BaseModel):
    label: str
    probability: float = Field(..., ge=0.0, le=1.0)


class DiseaseInfo(BaseModel):
    class_name: Optional[str] = None
    description: Optional[str] = None
    common_symptoms: Optional[str] = None
    severity_level: Optional[str] = None
    recommended_action: Optional[str] = None
    is_contagious: Optional[Any] = None
    related_conditions: Optional[str] = None
    image_indicators: Optional[str] = None


class AnalyzeResponse(BaseModel):
    status: str
    stage: str
    pipeline: list[str]
    final_class: Optional[str] = None
    model_prediction: Optional[BinaryPrediction] = None
    binary_prediction: Optional[BinaryPrediction] = None
    saved_image: Optional[str] = None
    saved_json: Optional[str] = None
    disease_info: Optional[DiseaseInfo] = None
    details: Optional[dict] = None


# ─────────────────────────────────────────────
#  CUSTOM EXCEPTIONS
# ─────────────────────────────────────────────
class ImageCorruptError(Exception):
    """Raised when the uploaded image fails corruption checks."""


class ImageQualityError(Exception):
    """Raised when the uploaded image is too blurry / low-quality."""


class ImageReadError(Exception):
    """Raised when OpenCV cannot decode the image."""


class ModelInferenceError(Exception):
    """Raised when a model prediction fails unexpectedly."""


# ─────────────────────────────────────────────
#  DISEASE CACHE  (singleton, loads once)
# ─────────────────────────────────────────────
class DiseaseCache:
    """Thread-safe, in-RAM lookup table built from SQLite at startup."""

    _instance: Optional["DiseaseCache"] = None

    def __new__(cls) -> "DiseaseCache":
        if cls._instance is None:
            cls._instance = super().__new__(cls)
            cls._instance._cache: dict[str, dict] = {}
            cls._instance._loaded = False
        return cls._instance

    def load(self) -> None:
        if self._loaded:
            return
        t0 = time.perf_counter()
        conn = sqlite3.connect(config.DB_PATH)
        cur = conn.cursor()
        cur.execute(
            """SELECT class_name, description, common_symptoms, severity_level,
                      recommended_action, is_contagious, related_conditions, image_indicators
               FROM diseases"""
        )
        for row in cur.fetchall():
            self._cache[row[0].lower()] = {
                "class_name": row[0],
                "description": row[1],
                "common_symptoms": row[2],
                "severity_level": row[3],
                "recommended_action": row[4],
                "is_contagious": row[5],
                "related_conditions": row[6],
                "image_indicators": row[7],
            }
        conn.close()
        elapsed = (time.perf_counter() - t0) * 1000
        log.info(f"[green]✔ Disease cache loaded[/green] — {len(self._cache)} entries in [bold]{elapsed:.1f}ms[/bold]")
        self._loaded = True

    def get(self, name: str) -> Optional[dict]:
        return self._cache.get(name.lower()) if name else None


# ─────────────────────────────────────────────
#  ML MODEL REGISTRY
# ─────────────────────────────────────────────
class ModelRegistry:
    """Holds all ML models; loaded once at startup."""

    def __init__(self) -> None:
        self.binary_model = None
        self.stage2_model = None

    def load_all(self) -> None:
        t0 = time.perf_counter()
        log.info("[yellow]⟳ Loading and warming up ML models …[/yellow]")
        self.binary_model = load_keras_model(config.BINARY_MODEL_PATH)
        self.stage2_model = load_keras_model(config.MODEL_A_PATH)

        # Trigger load
        b_model = self.binary_model.load()
        s_model = self.stage2_model.load()

        # Warmup (First inference is always slow in TF)
        log.info("[blue]⟳ Warming up models…[/blue]")
        dummy_bin = np.zeros((1, 224, 224, 3), dtype=np.float32)
        dummy_s2 = np.zeros((1, 300, 300, 3), dtype=np.float32)
        b_model.predict(dummy_bin, verbose=0)
        s_model.predict(dummy_s2, verbose=0)

        elapsed = (time.perf_counter() - t0) * 1000
        log.info(f"[green]✔ Models ready[/green] in [bold]{elapsed:.1f}ms[/bold]")


# ─────────────────────────────────────────────
#  EXECUTOR POOL MANAGER
# ─────────────────────────────────────────────
class PoolManager:
    """Creates and tears down thread/process pools."""

    def __init__(self) -> None:
        cpu = max(1, multiprocessing.cpu_count())
        self.predict_pool = ThreadPoolExecutor(max_workers=max(2, cpu - 1), thread_name_prefix="predict")
        self.save_pool = ProcessPoolExecutor(max_workers=max(1, cpu // 2))
        log.info(
            f"[green]✔ Pools ready[/green] — "
            f"predict threads=[bold]{max(2, cpu-1)}[/bold]  "
            f"save procs=[bold]{max(1, cpu//2)}[/bold]"
        )

    def shutdown(self) -> None:
        self.predict_pool.shutdown(wait=False)
        self.save_pool.shutdown(wait=False)
        log.info("[red]✖ Executor pools shut down[/red]")


# ─────────────────────────────────────────────
#  PIPELINE CORE
# ─────────────────────────────────────────────
class InferencePipeline:
    """
    Orchestrates the full image-→-diagnosis pipeline.
    Pure functions; state is injected via constructor.
    """

    BINARY_SIZE: int = 224
    STAGE2_SIZE: int = 300

    def __init__(
        self,
        models: ModelRegistry,
        pools: PoolManager,
        cache: DiseaseCache,
        result_dir: Path,
    ) -> None:
        self.models = models
        self.pools = pools
        self.cache = cache
        self.result_dir = result_dir

    # ── helpers ──────────────────────────────
    @staticmethod
    def _read_image(path: str) -> np.ndarray:
        """Decode image to BGR numpy array robustly (unicode-safe)."""
        bgr = cv2.imdecode(np.fromfile(path, dtype=np.uint8), cv2.IMREAD_COLOR)
        if bgr is None:
            bgr = cv2.imread(path, cv2.IMREAD_COLOR)
        if bgr is None:
            raise ImageReadError(f"OpenCV could not decode: {path}")
        return bgr

    @staticmethod
    def _save_outputs(save_dir: str, base_name: str, img_bgr: np.ndarray, metadata: dict) -> tuple[str, str]:
        """Write JPEG + JSON to disk (runs inside process pool)."""
        try:
            img_path = os.path.join(save_dir, base_name + ".jpg")
            _, enc = cv2.imencode(".jpg", img_bgr, [int(cv2.IMWRITE_JPEG_QUALITY), 92])
            Path(img_path).write_bytes(enc.tobytes())

            json_path = os.path.join(save_dir, base_name + ".json")
            with open(json_path, "w", encoding="utf-8") as jf:
                json.dump(metadata, jf, indent=2, ensure_ascii=False)

            return img_path, json_path
        except Exception:
            return "", ""

    # ── main pipeline ────────────────────────
    async def run(self, saved_path: str) -> AnalyzeResponse:
        loop = asyncio.get_running_loop()
        pipeline: list[str] = ["Image uploaded"]
        t_total = time.perf_counter()

        # 1 & 2) Fast Validation ──────────────
        pipeline.append("Validating image")
        t_val = time.perf_counter()
        
        val_info = await loop.run_in_executor(
            self.pools.predict_pool, 
            validate_image, 
            saved_path
        )
        
        elapsed_val = (time.perf_counter() - t_val) * 1000
        log.debug(f"Validation took {elapsed_val:.1f}ms")

        if not val_info.get("overall_ok", False):
            pipeline.append(f"Validation failed: {val_info.get('error')}")
            self._log_pipeline(pipeline, stage="invalid")
            return AnalyzeResponse(
                status="ok", stage="invalid", pipeline=pipeline, details=val_info
            )
        
        lap_var = val_info.get("laplacian_variance", 0)
        pipeline.append(f"Validation passed (var={lap_var:.1f})")

        # 3) Read pixels (Once) ────────────────
        img_bgr = await loop.run_in_executor(None, self._read_image, saved_path)

        # 4 & 5) Parallel Model Inference ─────
        # We run both in parallel to minimize latency. 
        # Even if binary says 'not_skin', running Stage-2 concurrently saves time for valid cases.
        pipeline.append("Running dual-model inference (parallel)")
        t_models = time.perf_counter()

        task_binary = loop.run_in_executor(
            self.pools.predict_pool,
            predict_binary,
            self.models.binary_model,
            img_bgr,
            self.BINARY_SIZE,
        )
        task_stage2 = loop.run_in_executor(
            self.pools.predict_pool,
            predict_stage2,
            self.models.stage2_model,
            img_bgr,
            self.STAGE2_SIZE,
        )

        (bin_label, bin_prob), (idx, prob) = await asyncio.gather(task_binary, task_stage2)
        
        elapsed_models = (time.perf_counter() - t_models) * 1000
        log.debug(f"Dual-inference took {elapsed_models:.1f}ms")

        pipeline.append(f"Binary: {bin_label} ({bin_prob:.2f})")
        
        if bin_label == "not_skin":
            self._log_pipeline(pipeline, stage="not_skin")
            return AnalyzeResponse(
                status="ok",
                stage="not_skin",
                pipeline=pipeline,
                binary_prediction=BinaryPrediction(label="not_skin", probability=bin_prob),
            )

        # Process Stage-2 results
        final_class = config.STAGE2_CLASS_NAMES[idx]
        pipeline.append(f"Final → {final_class} ({prob:.3f})")

        # 6) DB lookup (RAM cache, instant) ───
        info_raw = self.cache.get(final_class)
        disease_info = DiseaseInfo(**(info_raw or {})) if info_raw else None
        pipeline.append("Disease info loaded from cache")

        # 7) Async save (Background ready) ────
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        base_name = f"{final_class.replace(' ', '_')}_{int(prob * 100)}_{timestamp}"
        
        # Pre-calculate paths so we can return them immediately
        img_path = str(self.result_dir / f"{base_name}.jpg")
        json_path = str(self.result_dir / f"{base_name}.json")
        
        metadata = {
            "timestamp": timestamp,
            "predicted_class": final_class,
            "model_confidence": prob,
            "binary_confidence": bin_prob,
            "pipeline": pipeline,
            "disease_info": info_raw,
        }
        
        # Arguments for the background save task
        save_args = (str(self.result_dir), base_name, img_bgr, metadata)

        elapsed_ms = (time.perf_counter() - t_total) * 1000
        self._log_pipeline(pipeline, stage="disease_found", elapsed_ms=elapsed_ms, final_class=final_class, prob=prob)

        return AnalyzeResponse(
            status="ok",
            stage="disease_found",
            pipeline=pipeline,
            final_class=final_class,
            model_prediction=BinaryPrediction(label=final_class, probability=prob),
            binary_prediction=BinaryPrediction(label="skin", probability=bin_prob),
            saved_image=img_path,
            saved_json=json_path,
            disease_info=disease_info,
            details={"save_args": save_args}
        )

    def _submit_save(self, base_name: str, img_bgr: np.ndarray, metadata: dict) -> tuple[str, str]:
        """Submits save to process pool and immediately waits (max 10s)."""
        future = self.pools.save_pool.submit(
            self._save_outputs, str(self.result_dir), base_name, img_bgr, metadata
        )
        try:
            return future.result(timeout=10)
        except Exception:
            log.warning("[yellow]⚠ Async save timed out or failed[/yellow]")
            return "", ""

    @staticmethod
    def _log_pipeline(
        pipeline: list[str],
        stage: str,
        elapsed_ms: float = 0.0,
        final_class: str = "",
        prob: float = 0.0,
    ) -> None:
        table = Table(box=box.SIMPLE_HEAVY, show_header=True, header_style="bold cyan")
        table.add_column("#", style="dim", width=3)
        table.add_column("Step", style="white")

        ICONS = {0: "📤", 1: "🔍", 2: "📊", 3: "🧠", 4: "🔬", 5: "💾"}
        for i, step in enumerate(pipeline):
            table.add_row(str(i + 1), step)

        stage_color = {"disease_found": "green", "not_skin": "yellow", "invalid": "red"}.get(stage, "white")
        summary = (
            f"[bold {stage_color}]{stage.upper()}[/bold {stage_color}]"
            + (f"  ·  [bold]{final_class}[/bold] ({prob:.1%})" if final_class else "")
            + (f"  ·  [dim]{elapsed_ms:.0f}ms[/dim]" if elapsed_ms else "")
        )
        console.print(Panel(table, title=summary, border_style=stage_color, expand=False))


# ─────────────────────────────────────────────
#  APP FACTORY (Singletons initialized in lifespan)
# ─────────────────────────────────────────────
models: ModelRegistry = None
pools: PoolManager = None
cache: DiseaseCache = None
pipeline_runner: Optional[InferencePipeline] = None

UPLOAD_FOLDER = Path(config.UPLOAD_FOLDER)
RESULT_SAVE_DIR = Path(__file__).parent / "uploaded_images"
UPLOAD_FOLDER.mkdir(parents=True, exist_ok=True)
RESULT_SAVE_DIR.mkdir(parents=True, exist_ok=True)


@asynccontextmanager
async def lifespan(application: FastAPI):
    """Startup → run app → Shutdown."""
    console.print(
        Panel.fit(
            "[bold cyan]🩺  Skin Disease Detection API[/bold cyan]\n"
            "[dim]FastAPI · Async · EfficientNetV2-M[/dim]",
            border_style="cyan",
        )
    )
    global models, pools, cache, pipeline_runner

    # Initialize singletons
    cache = DiseaseCache()
    cache.load()

    models = ModelRegistry()
    models.load_all()

    pools = PoolManager()

    pipeline_runner = InferencePipeline(models, pools, cache, RESULT_SAVE_DIR)

    log.info("[bold green]🚀  Server ready — listening for requests[/bold green]")
    yield
    if pools:
        pools.shutdown()
    log.info("[red]Server stopped.[/red]")


# ─────────────────────────────────────────────
#  FAST API APP
# ─────────────────────────────────────────────
app = FastAPI(
    title="Skin Disease Detection API",
    version="2.0.0",
    description="High-speed async skin disease classifier powered by EfficientNetV2-M.",
    lifespan=lifespan,
    docs_url="/docs",
    redoc_url="/redoc",
)

# Ensure directories exist before mounting
os.makedirs("static", exist_ok=True)
os.makedirs("templates", exist_ok=True)

app.mount("/static", StaticFiles(directory="static"), name="static")
templates = Jinja2Templates(directory="templates")


# ─────────────────────────────────────────────
#  MIDDLEWARE: request timing log
# ─────────────────────────────────────────────
@app.middleware("http")
async def log_requests(request: Request, call_next):
    t0 = time.perf_counter()
    response = await call_next(request)
    elapsed = (time.perf_counter() - t0) * 1000
    color = "green" if response.status_code < 400 else "red"
    log.info(
        f"[{color}]{response.status_code}[/{color}]  "
        f"[bold]{request.method}[/bold] {request.url.path}  "
        f"[dim]{elapsed:.1f}ms[/dim]"
    )
    return response


# ─────────────────────────────────────────────
#  ROUTES
# ─────────────────────────────────────────────
@app.get("/", response_class=HTMLResponse, include_in_schema=False)
async def index(request: Request):
    return templates.TemplateResponse("index.html", {"request": request})


@app.get("/health", tags=["Utility"])
async def health():
    """Quick health-check endpoint."""
    return {
        "status": "ok",
        "disease_cache_entries": len(cache._cache) if cache else 0,
        "models_loaded": models and models.binary_model is not None and models.stage2_model is not None,
    }


@app.post(
    "/api/analyze",
    response_model=AnalyzeResponse,
    tags=["Inference"],
    summary="Analyze a skin image",
    status_code=status.HTTP_200_OK,
)
async def analyze(
    background_tasks: BackgroundTasks,
    image: UploadFile = File(..., description="JPG/PNG skin image"),
):
    # ── basic validation ──────────────────────
    if not image.filename:
        raise HTTPException(status_code=400, detail="Empty filename.")

    ext = image.filename.rsplit(".", 1)[-1].lower()
    if ext not in {"jpg", "jpeg", "png", "bmp", "webp"}:
        raise HTTPException(status_code=415, detail=f"Unsupported image format: {ext}")

    # ── save upload to disk (non-blocking) ───
    uid = uuid.uuid4().hex
    saved_path = UPLOAD_FOLDER / f"{uid}.{ext}"
    try:
        contents = await image.read()
        if len(contents) == 0:
            raise HTTPException(status_code=400, detail="Uploaded file is empty.")
        await asyncio.to_thread(saved_path.write_bytes, contents)
    except HTTPException:
        raise
    except Exception as exc:
        log.error(f"[red]File save failed:[/red] {exc}")
        raise HTTPException(status_code=500, detail="Could not save uploaded file.")

    # ── schedule temp file cleanup ────────────
    background_tasks.add_task(_cleanup_temp, saved_path)

    # ── run inference pipeline ────────────────
    try:
        result = await pipeline_runner.run(str(saved_path))
        
        # If the result includes save arguments, fire off the background save
        if result.details and "save_args" in result.details:
            s_args = result.details.pop("save_args")  # Remove from dict to avoid Pydantic serialization error
            background_tasks.add_task(
                pipeline_runner.pools.save_pool.submit, 
                pipeline_runner._save_outputs, 
                *s_args
            )
            
        return result
    except ImageReadError as exc:
        log.error(f"[red]ImageReadError:[/red] {exc}")
        raise HTTPException(status_code=422, detail=str(exc))
    except ModelInferenceError as exc:
        log.error(f"[red]ModelInferenceError:[/red] {exc}")
        raise HTTPException(status_code=500, detail=str(exc))
    except Exception as exc:
        log.error(f"[red]Unhandled pipeline error:[/red]\n{traceback.format_exc()}")
        raise HTTPException(status_code=500, detail=f"Internal server error: {exc}")


# ─────────────────────────────────────────────
#  BACKGROUND HELPERS
# ─────────────────────────────────────────────
async def _cleanup_temp(path: Path) -> None:
    """Delete temporary upload file after response is sent."""
    try:
        await asyncio.to_thread(path.unlink, missing_ok=True)
    except Exception:
        pass


# ─────────────────────────────────────────────
#  GLOBAL EXCEPTION HANDLER
# ─────────────────────────────────────────────
@app.exception_handler(Exception)
async def global_exception_handler(request: Request, exc: Exception):
    log.error(f"[red]Unhandled exception on {request.url}:[/red]\n{traceback.format_exc()}")
    return JSONResponse(
        status_code=500,
        content={"status": "error", "message": str(exc), "path": str(request.url)},
    )


# ─────────────────────────────────────────────
#  ENTRY POINT
# ─────────────────────────────────────────────
if __name__ == "__main__":
    uvicorn.run(
        app,                  # Pass app object directly to avoid double-import
        host="192.168.121.85",
        port=8001,
        reload=False,         # set True only in dev
        workers=1,            # >1 only with gunicorn; Keras models are not fork-safe
        loop="auto",
        http="auto",
        access_log=False,
        log_config=None,
    )