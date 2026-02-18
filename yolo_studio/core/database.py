"""Database module for YOLO Studio.

Defines all SQLAlchemy ORM models, enum types, and session-management helpers
used throughout the application.  The database is a single SQLite file stored
in the user's application-data directory (or the current working directory as a
fallback).

Tables
------
- Dataset          — registered YOLO-format dataset entries
- TrainingRun      — individual model training runs with hyperparameters & metrics
- RemoteDevice     — edge devices (Jetson / Xavier / Raspberry Pi) reachable via WebSocket
- RemoteTestResult — inference results returned from a remote device test run

Typical usage
-------------
    from core.database import init_db, get_session, Dataset

    init_db()                          # create tables if they don't exist

    with get_session() as session:
        datasets = session.query(Dataset).all()
"""

from __future__ import annotations

import enum
import json
import logging
import os
from contextlib import contextmanager
from datetime import datetime
from pathlib import Path
from typing import Generator, Optional

from sqlalchemy import (
    Boolean,
    DateTime,
    Enum,
    Float,
    ForeignKey,
    Integer,
    String,
    Text,
    create_engine,
    event,
)
from sqlalchemy.orm import (
    DeclarativeBase,
    Mapped,
    mapped_column,
    relationship,
    scoped_session,
    sessionmaker,
)
from sqlalchemy.types import TypeDecorator

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Database path resolution
# ---------------------------------------------------------------------------


def get_db_path() -> Path:
    """Return the absolute path to the SQLite database file.

    The path is resolved in the following priority order:

    1. ``YOLO_STUDIO_DB`` environment variable (useful for testing).
    2. A platform-appropriate user-data directory:
       - Linux/macOS: ``~/.local/share/yolo_studio/yolo_studio.db``
       - Windows:     ``%APPDATA%\\yolo_studio\\yolo_studio.db``
    3. ``./yolo_studio.db`` in the current working directory as a final fallback.

    Returns:
        Path: Absolute path to the SQLite database file.
    """
    env_override = os.environ.get("YOLO_STUDIO_DB")
    if env_override:
        return Path(env_override).expanduser().resolve()

    if os.name == "nt":
        base = Path(os.environ.get("APPDATA", Path.home())) / "yolo_studio"
    else:
        base = Path.home() / ".local" / "share" / "yolo_studio"

    base.mkdir(parents=True, exist_ok=True)
    return base / "yolo_studio.db"


# ---------------------------------------------------------------------------
# Custom column types
# ---------------------------------------------------------------------------


class JSONList(TypeDecorator):
    """SQLAlchemy column type that transparently serialises Python lists to/from JSON.

    Stored as TEXT in SQLite.  Handles ``None`` values gracefully by returning
    an empty list on read.

    Example::

        class_names: Mapped[list] = mapped_column(JSONList, default=list)
    """

    impl = Text
    cache_ok = True

    def process_bind_param(self, value: Optional[list], dialect) -> Optional[str]:
        """Serialise a Python list to a JSON string before writing to the DB.

        Args:
            value: The Python list (or ``None``) to serialise.
            dialect: The SQLAlchemy dialect in use (unused).

        Returns:
            A JSON-encoded string, or ``None`` if *value* is ``None``.
        """
        if value is None:
            return None
        return json.dumps(value)

    def process_result_value(self, value: Optional[str], dialect) -> list:
        """Deserialise a JSON string from the DB back into a Python list.

        Args:
            value: The raw JSON string from SQLite (or ``None``).
            dialect: The SQLAlchemy dialect in use (unused).

        Returns:
            A Python list; returns ``[]`` when *value* is ``None`` or empty.
        """
        if not value:
            return []
        try:
            return json.loads(value)
        except (json.JSONDecodeError, TypeError):
            logger.warning("Failed to decode JSON list from DB: %r", value)
            return []


class JSONDict(TypeDecorator):
    """SQLAlchemy column type that transparently serialises Python dicts to/from JSON.

    Stored as TEXT in SQLite.  Handles ``None`` values gracefully by returning
    an empty dict on read.

    Example::

        config_yaml: Mapped[dict] = mapped_column(JSONDict, default=dict)
    """

    impl = Text
    cache_ok = True

    def process_bind_param(self, value: Optional[dict], dialect) -> Optional[str]:
        """Serialise a Python dict to a JSON string before writing to the DB.

        Args:
            value: The Python dict (or ``None``) to serialise.
            dialect: The SQLAlchemy dialect in use (unused).

        Returns:
            A JSON-encoded string, or ``None`` if *value* is ``None``.
        """
        if value is None:
            return None
        return json.dumps(value)

    def process_result_value(self, value: Optional[str], dialect) -> dict:
        """Deserialise a JSON string from the DB back into a Python dict.

        Args:
            value: The raw JSON string from SQLite (or ``None``).
            dialect: The SQLAlchemy dialect in use (unused).

        Returns:
            A Python dict; returns ``{}`` when *value* is ``None`` or empty.
        """
        if not value:
            return {}
        try:
            return json.loads(value)
        except (json.JSONDecodeError, TypeError):
            logger.warning("Failed to decode JSON dict from DB: %r", value)
            return {}


# ---------------------------------------------------------------------------
# Enum definitions
# ---------------------------------------------------------------------------


class DatasetSource(str, enum.Enum):
    """Origin of a dataset entry.

    Attributes:
        MANUAL:      Created manually by the user inside YOLO Studio.
        ROBOFLOW:    Downloaded from Roboflow Universe via the Discover tab.
        HUGGINGFACE: Downloaded from HuggingFace Hub via the Discover tab.
    """

    MANUAL = "manual"
    ROBOFLOW = "roboflow"
    HUGGINGFACE = "huggingface"


class TrainingStatus(str, enum.Enum):
    """Lifecycle status of a training run.

    Attributes:
        PENDING:   Run has been created but not yet started.
        RUNNING:   Training is actively in progress.
        COMPLETED: Training finished successfully.
        FAILED:    Training exited with an error.
        ARCHIVED:  Run has been soft-deleted / archived by the user.
    """

    PENDING = "pending"
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"
    ARCHIVED = "archived"


class DeviceType(str, enum.Enum):
    """Hardware type of a registered remote edge device.

    Attributes:
        JETSON_NANO:  NVIDIA Jetson Nano.
        XAVIER:       NVIDIA Jetson AGX / NX Xavier.
        RASPBERRY_PI: Raspberry Pi (any model).
        OTHER:        Any other device running the edge agent.
    """

    JETSON_NANO = "jetson_nano"
    XAVIER = "xavier"
    RASPBERRY_PI = "raspberry_pi"
    OTHER = "other"


class DeviceStatus(str, enum.Enum):
    """Last-known connectivity status of a remote device.

    Attributes:
        ONLINE:  Device responded to the most recent ping.
        OFFLINE: Device did not respond or connection was refused.
        UNKNOWN: Device has never been pinged in this session.
    """

    ONLINE = "online"
    OFFLINE = "offline"
    UNKNOWN = "unknown"


# ---------------------------------------------------------------------------
# ORM declarative base
# ---------------------------------------------------------------------------


class Base(DeclarativeBase):
    """Shared declarative base for all YOLO Studio ORM models."""

    pass


# ---------------------------------------------------------------------------
# Model: Dataset
# ---------------------------------------------------------------------------


class Dataset(Base):
    """ORM model representing a registered YOLO-format dataset.

    Each row corresponds to one dataset folder on disk that has been imported
    into YOLO Studio — either manually, from Roboflow Universe, or from
    HuggingFace Hub.

    Attributes:
        id:                  Auto-increment primary key.
        name:                Human-readable dataset name.
        description:         Optional longer description.
        created_at:          Timestamp when the record was first created.
        updated_at:          Timestamp of the most recent update.
        source:              Where the dataset came from (see :class:`DatasetSource`).
        roboflow_project_id: Roboflow project slug, e.g.
                             ``"workspace/project/version"``.
        hf_repo_id:          HuggingFace repository ID, e.g.
                             ``"username/dataset-name"``.
        local_path:          Absolute path to the dataset root folder on disk.
        class_names:         Ordered list of class label strings (JSON array).
        num_images:          Total image count across all splits.
        num_classes:         Number of distinct classes.
        tags:                Free-form tag list for searching/filtering (JSON array).
        training_runs:       Back-reference to all :class:`TrainingRun` rows that
                             use this dataset.
    """

    __tablename__ = "datasets"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    name: Mapped[str] = mapped_column(String(255), nullable=False)
    description: Mapped[Optional[str]] = mapped_column(Text, nullable=True)
    created_at: Mapped[datetime] = mapped_column(
        DateTime, nullable=False, default=datetime.utcnow
    )
    updated_at: Mapped[datetime] = mapped_column(
        DateTime, nullable=False, default=datetime.utcnow, onupdate=datetime.utcnow
    )
    source: Mapped[str] = mapped_column(
        Enum(DatasetSource), nullable=False, default=DatasetSource.MANUAL
    )
    roboflow_project_id: Mapped[Optional[str]] = mapped_column(
        String(512), nullable=True
    )
    hf_repo_id: Mapped[Optional[str]] = mapped_column(String(512), nullable=True)
    local_path: Mapped[str] = mapped_column(String(1024), nullable=False, default="")
    class_names: Mapped[list] = mapped_column(JSONList, nullable=False, default=list)
    num_images: Mapped[int] = mapped_column(Integer, nullable=False, default=0)
    num_classes: Mapped[int] = mapped_column(Integer, nullable=False, default=0)
    tags: Mapped[list] = mapped_column(JSONList, nullable=False, default=list)

    # Relationships
    training_runs: Mapped[list["TrainingRun"]] = relationship(
        "TrainingRun", back_populates="dataset", cascade="all, delete-orphan"
    )

    def __repr__(self) -> str:
        return (
            f"<Dataset id={self.id} name={self.name!r} "
            f"source={self.source} classes={self.num_classes}>"
        )

    def to_dict(self) -> dict:
        """Serialise this dataset record to a plain Python dictionary.

        Returns:
            A dict with all column values, suitable for JSON serialisation.
            Datetime fields are converted to ISO-8601 strings.
        """
        return {
            "id": self.id,
            "name": self.name,
            "description": self.description,
            "created_at": self.created_at.isoformat() if self.created_at else None,
            "updated_at": self.updated_at.isoformat() if self.updated_at else None,
            "source": self.source,
            "roboflow_project_id": self.roboflow_project_id,
            "hf_repo_id": self.hf_repo_id,
            "local_path": self.local_path,
            "class_names": self.class_names,
            "num_images": self.num_images,
            "num_classes": self.num_classes,
            "tags": self.tags,
        }


# ---------------------------------------------------------------------------
# Model: TrainingRun
# ---------------------------------------------------------------------------


class TrainingRun(Base):
    """ORM model representing a single YOLO model training run.

    Stores both the configuration used to launch the run *and* the resulting
    metrics once training completes.  Long-running operations update the row
    in-place as the run progresses.

    Attributes:
        id:                   Auto-increment primary key.
        name:                 User-supplied run label.
        notes:                Optional free-text notes about this run.
        created_at:           Timestamp when the run record was created.
        completed_at:         Timestamp when training finished (``None`` if still running).
        dataset_id:           FK → :class:`Dataset` used for training.
        model_architecture:   Model variant string, e.g. ``"yolov8n"`` or ``"yolov11m"``.
        image_size:           Input image resolution (square), e.g. ``640``.
        batch_size:           Images per gradient-descent step.
        epochs:               Total training epochs requested.
        learning_rate:        Initial learning rate.
        optimizer:            Optimiser name, e.g. ``"SGD"`` or ``"AdamW"``.
        warmup_epochs:        Number of warmup epochs for the LR scheduler.
        weight_decay:         L2 regularisation coefficient.
        mosaic:               Mosaic augmentation enabled flag.
        mixup:                Mixup augmentation enabled flag.
        copy_paste:           Copy-paste augmentation enabled flag.
        hsv_augment:          HSV colour-space augmentation enabled flag.
        flip_augment:         Random horizontal-flip augmentation enabled flag.
        pretrained:           Whether Ultralytics default pretrained weights are used.
        custom_weights_path:  Path to a custom ``.pt`` file used as starting weights.
        status:               Lifecycle status (see :class:`TrainingStatus`).
        best_map50:           Best mAP\@0.5 metric achieved during training.
        best_map50_95:        Best mAP\@0.5:0.95 metric achieved during training.
        final_loss:           Final combined training loss value.
        output_dir:           Path to the Ultralytics ``runs/train/expN`` directory.
        weights_path:         Path to the ``best.pt`` weights file.
        is_saved:             ``True`` once the user clicks "Save This Run".
        config_yaml:          Full hyperparameter snapshot as a JSON dict.
        dataset:              Relationship to the parent :class:`Dataset`.
        remote_test_results:  Back-reference to :class:`RemoteTestResult` rows
                              for this run.
    """

    __tablename__ = "training_runs"

    # ── Identity ─────────────────────────────────────────────────────────────
    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    name: Mapped[str] = mapped_column(String(255), nullable=False)
    notes: Mapped[Optional[str]] = mapped_column(Text, nullable=True)
    created_at: Mapped[datetime] = mapped_column(
        DateTime, nullable=False, default=datetime.utcnow
    )
    completed_at: Mapped[Optional[datetime]] = mapped_column(DateTime, nullable=True)

    # ── Dataset reference ────────────────────────────────────────────────────
    dataset_id: Mapped[Optional[int]] = mapped_column(
        Integer, ForeignKey("datasets.id", ondelete="SET NULL"), nullable=True
    )

    # ── Architecture & data ──────────────────────────────────────────────────
    model_architecture: Mapped[str] = mapped_column(
        String(64), nullable=False, default="yolov8n"
    )
    image_size: Mapped[int] = mapped_column(Integer, nullable=False, default=640)
    batch_size: Mapped[int] = mapped_column(Integer, nullable=False, default=16)
    epochs: Mapped[int] = mapped_column(Integer, nullable=False, default=100)
    learning_rate: Mapped[float] = mapped_column(Float, nullable=False, default=0.01)
    optimizer: Mapped[str] = mapped_column(String(32), nullable=False, default="SGD")
    warmup_epochs: Mapped[float] = mapped_column(Float, nullable=False, default=3.0)
    weight_decay: Mapped[float] = mapped_column(Float, nullable=False, default=0.0005)

    # ── Augmentation flags ───────────────────────────────────────────────────
    mosaic: Mapped[bool] = mapped_column(Boolean, nullable=False, default=True)
    mixup: Mapped[bool] = mapped_column(Boolean, nullable=False, default=False)
    copy_paste: Mapped[bool] = mapped_column(Boolean, nullable=False, default=False)
    hsv_augment: Mapped[bool] = mapped_column(Boolean, nullable=False, default=True)
    flip_augment: Mapped[bool] = mapped_column(Boolean, nullable=False, default=True)

    # ── Weights ──────────────────────────────────────────────────────────────
    pretrained: Mapped[bool] = mapped_column(Boolean, nullable=False, default=True)
    custom_weights_path: Mapped[Optional[str]] = mapped_column(
        String(1024), nullable=True
    )

    # ── Status & metrics ─────────────────────────────────────────────────────
    status: Mapped[str] = mapped_column(
        Enum(TrainingStatus), nullable=False, default=TrainingStatus.PENDING
    )
    best_map50: Mapped[Optional[float]] = mapped_column(Float, nullable=True)
    best_map50_95: Mapped[Optional[float]] = mapped_column(Float, nullable=True)
    final_loss: Mapped[Optional[float]] = mapped_column(Float, nullable=True)

    # ── Output paths ─────────────────────────────────────────────────────────
    output_dir: Mapped[Optional[str]] = mapped_column(String(1024), nullable=True)
    weights_path: Mapped[Optional[str]] = mapped_column(String(1024), nullable=True)

    # ── User flags ───────────────────────────────────────────────────────────
    is_saved: Mapped[bool] = mapped_column(Boolean, nullable=False, default=False)

    # ── Hyperparameter snapshot ───────────────────────────────────────────────
    config_yaml: Mapped[dict] = mapped_column(JSONDict, nullable=False, default=dict)

    # ── Relationships ─────────────────────────────────────────────────────────
    dataset: Mapped[Optional["Dataset"]] = relationship(
        "Dataset", back_populates="training_runs"
    )
    remote_test_results: Mapped[list["RemoteTestResult"]] = relationship(
        "RemoteTestResult", back_populates="training_run", cascade="all, delete-orphan"
    )

    def __repr__(self) -> str:
        return (
            f"<TrainingRun id={self.id} name={self.name!r} "
            f"arch={self.model_architecture} status={self.status}>"
        )

    def to_dict(self) -> dict:
        """Serialise this training run to a plain Python dictionary.

        Returns:
            A dict with all column values.  Datetime fields are ISO-8601 strings.
        """
        return {
            "id": self.id,
            "name": self.name,
            "notes": self.notes,
            "created_at": self.created_at.isoformat() if self.created_at else None,
            "completed_at": (
                self.completed_at.isoformat() if self.completed_at else None
            ),
            "dataset_id": self.dataset_id,
            "model_architecture": self.model_architecture,
            "image_size": self.image_size,
            "batch_size": self.batch_size,
            "epochs": self.epochs,
            "learning_rate": self.learning_rate,
            "optimizer": self.optimizer,
            "warmup_epochs": self.warmup_epochs,
            "weight_decay": self.weight_decay,
            "mosaic": self.mosaic,
            "mixup": self.mixup,
            "copy_paste": self.copy_paste,
            "hsv_augment": self.hsv_augment,
            "flip_augment": self.flip_augment,
            "pretrained": self.pretrained,
            "custom_weights_path": self.custom_weights_path,
            "status": self.status,
            "best_map50": self.best_map50,
            "best_map50_95": self.best_map50_95,
            "final_loss": self.final_loss,
            "output_dir": self.output_dir,
            "weights_path": self.weights_path,
            "is_saved": self.is_saved,
            "config_yaml": self.config_yaml,
        }

    def build_config_snapshot(self) -> dict:
        """Build a hyperparameter snapshot dict from the current column values.

        Populate :attr:`config_yaml` with this just before launching a training
        run so that the exact configuration is preserved for reproducibility.

        Returns:
            A dict of every training hyperparameter keyed by Ultralytics
            argument name where applicable.
        """
        return {
            "model_architecture": self.model_architecture,
            "image_size": self.image_size,
            "batch_size": self.batch_size,
            "epochs": self.epochs,
            "learning_rate": self.learning_rate,
            "optimizer": self.optimizer,
            "warmup_epochs": self.warmup_epochs,
            "weight_decay": self.weight_decay,
            "mosaic": self.mosaic,
            "mixup": self.mixup,
            "copy_paste": self.copy_paste,
            "hsv_augment": self.hsv_augment,
            "flip_augment": self.flip_augment,
            "pretrained": self.pretrained,
            "custom_weights_path": self.custom_weights_path,
        }


# ---------------------------------------------------------------------------
# Model: RemoteDevice
# ---------------------------------------------------------------------------


class RemoteDevice(Base):
    """ORM model representing a registered remote edge inference device.

    Stores the connection parameters needed to reach a device running
    ``edge/jetson_agent.py`` over WebSocket.

    Attributes:
        id:           Auto-increment primary key.
        name:         Human-readable label given by the user.
        device_type:  Hardware category (see :class:`DeviceType`).
        host:         Hostname or IP address of the device.
        port:         WebSocket port (typically ``8765``).
        auth_token:   Shared secret for authenticating messages.
        last_seen:    Timestamp of the most recent successful ping (or ``None``).
        status:       Last-known connectivity (see :class:`DeviceStatus`).
        test_results: Back-reference to all :class:`RemoteTestResult` rows from
                      this device.
    """

    __tablename__ = "remote_devices"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    name: Mapped[str] = mapped_column(String(255), nullable=False)
    device_type: Mapped[str] = mapped_column(
        Enum(DeviceType), nullable=False, default=DeviceType.OTHER
    )
    host: Mapped[str] = mapped_column(String(255), nullable=False)
    port: Mapped[int] = mapped_column(Integer, nullable=False, default=8765)
    auth_token: Mapped[str] = mapped_column(String(512), nullable=False)
    last_seen: Mapped[Optional[datetime]] = mapped_column(DateTime, nullable=True)
    status: Mapped[str] = mapped_column(
        Enum(DeviceStatus), nullable=False, default=DeviceStatus.UNKNOWN
    )

    # Relationships
    test_results: Mapped[list["RemoteTestResult"]] = relationship(
        "RemoteTestResult", back_populates="device", cascade="all, delete-orphan"
    )

    def __repr__(self) -> str:
        return (
            f"<RemoteDevice id={self.id} name={self.name!r} "
            f"host={self.host}:{self.port} status={self.status}>"
        )

    @property
    def ws_uri(self) -> str:
        """Construct the WebSocket URI for connecting to this device.

        Returns:
            A ``ws://`` URI string, e.g. ``"ws://192.168.1.50:8765"``.
        """
        return f"ws://{self.host}:{self.port}"

    def to_dict(self) -> dict:
        """Serialise this device record to a plain Python dictionary.

        Note:
            ``auth_token`` is included so the UI can pass it to the WebSocket
            client.  Treat this dict as sensitive — do not log it verbatim.

        Returns:
            A dict with all column values.  Datetime fields are ISO-8601 strings.
        """
        return {
            "id": self.id,
            "name": self.name,
            "device_type": self.device_type,
            "host": self.host,
            "port": self.port,
            "auth_token": self.auth_token,
            "last_seen": self.last_seen.isoformat() if self.last_seen else None,
            "status": self.status,
            "ws_uri": self.ws_uri,
        }


# ---------------------------------------------------------------------------
# Model: RemoteTestResult
# ---------------------------------------------------------------------------


class RemoteTestResult(Base):
    """ORM model storing the results of a remote inference test run.

    Created when the user clicks "Deploy & Test" in the Remote Devices tab.
    The associated device runs inference and streams metrics back; this record
    persists the final summary.

    Attributes:
        id:                 Auto-increment primary key.
        device_id:          FK → :class:`RemoteDevice` that ran inference.
        training_run_id:    FK → :class:`TrainingRun` whose weights were deployed.
        run_at:             Timestamp when the remote test was executed.
        test_dataset_path:  Path to the dataset used on the remote device.
        num_images_tested:  Number of images processed during the test.
        map50:              mAP\@0.5 score from remote inference.
        map50_95:           mAP\@0.5:0.95 score from remote inference.
        precision:          Precision score.
        recall:             Recall score.
        output_images_dir:  Directory where annotated result images are stored.
        notes:              Optional free-text notes added by the user.
        device:             Relationship to the parent :class:`RemoteDevice`.
        training_run:       Relationship to the parent :class:`TrainingRun`.
    """

    __tablename__ = "remote_test_results"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    device_id: Mapped[Optional[int]] = mapped_column(
        Integer, ForeignKey("remote_devices.id", ondelete="SET NULL"), nullable=True
    )
    training_run_id: Mapped[Optional[int]] = mapped_column(
        Integer, ForeignKey("training_runs.id", ondelete="SET NULL"), nullable=True
    )
    run_at: Mapped[datetime] = mapped_column(
        DateTime, nullable=False, default=datetime.utcnow
    )
    test_dataset_path: Mapped[Optional[str]] = mapped_column(
        String(1024), nullable=True
    )
    num_images_tested: Mapped[int] = mapped_column(Integer, nullable=False, default=0)
    map50: Mapped[Optional[float]] = mapped_column(Float, nullable=True)
    map50_95: Mapped[Optional[float]] = mapped_column(Float, nullable=True)
    precision: Mapped[Optional[float]] = mapped_column(Float, nullable=True)
    recall: Mapped[Optional[float]] = mapped_column(Float, nullable=True)
    output_images_dir: Mapped[Optional[str]] = mapped_column(
        String(1024), nullable=True
    )
    notes: Mapped[Optional[str]] = mapped_column(Text, nullable=True)

    # Relationships
    device: Mapped[Optional["RemoteDevice"]] = relationship(
        "RemoteDevice", back_populates="test_results"
    )
    training_run: Mapped[Optional["TrainingRun"]] = relationship(
        "TrainingRun", back_populates="remote_test_results"
    )

    def __repr__(self) -> str:
        return (
            f"<RemoteTestResult id={self.id} device_id={self.device_id} "
            f"run_id={self.training_run_id} map50={self.map50}>"
        )

    def to_dict(self) -> dict:
        """Serialise this test result to a plain Python dictionary.

        Returns:
            A dict with all column values.  Datetime fields are ISO-8601 strings.
        """
        return {
            "id": self.id,
            "device_id": self.device_id,
            "training_run_id": self.training_run_id,
            "run_at": self.run_at.isoformat() if self.run_at else None,
            "test_dataset_path": self.test_dataset_path,
            "num_images_tested": self.num_images_tested,
            "map50": self.map50,
            "map50_95": self.map50_95,
            "precision": self.precision,
            "recall": self.recall,
            "output_images_dir": self.output_images_dir,
            "notes": self.notes,
        }


# ---------------------------------------------------------------------------
# Engine & session factory
# ---------------------------------------------------------------------------

# Module-level singletons — populated by init_db().
_engine = None
_SessionFactory = None


def init_db(db_path: Optional[Path] = None) -> None:
    """Initialise the SQLAlchemy engine and create all tables.

    Must be called once at application startup before any call to
    :func:`get_session`.  Calling it more than once is safe — subsequent calls
    return immediately without re-initialising.

    Args:
        db_path: Optional explicit path to the SQLite file.  When omitted the
                 path from :func:`get_db_path` is used.

    Raises:
        sqlalchemy.exc.OperationalError: If the database file cannot be
            created or opened (e.g. permission denied).
    """
    global _engine, _SessionFactory

    if _engine is not None:
        # Already initialised — idempotent.
        return

    resolved_path = db_path or get_db_path()
    db_url = f"sqlite:///{resolved_path}"
    logger.info("Initialising database at: %s", resolved_path)

    _engine = create_engine(
        db_url,
        # SQLite requires this flag when the same connection is used across
        # multiple threads (as is the case with Qt's background threads).
        connect_args={"check_same_thread": False},
        echo=False,  # Flip to True for verbose SQL logging during development.
    )

    @event.listens_for(_engine, "connect")
    def _configure_pragmas(dbapi_conn, _connection_record):
        """Enable WAL mode and foreign-key enforcement on every new connection.

        Args:
            dbapi_conn: The raw DBAPI connection object.
            _connection_record: SQLAlchemy connection record (unused).
        """
        cursor = dbapi_conn.cursor()
        # WAL (Write-Ahead Log) journal mode gives significantly better
        # concurrent read/write performance than the default DELETE mode.
        cursor.execute("PRAGMA journal_mode=WAL")
        # SQLite does not enforce foreign keys by default — enable it here.
        cursor.execute("PRAGMA foreign_keys=ON")
        cursor.close()

    # Create all tables defined on Base subclasses that don't yet exist.
    Base.metadata.create_all(_engine)

    # scoped_session binds a separate Session to each thread automatically,
    # which is the correct pattern for a multi-threaded Qt application.
    _SessionFactory = scoped_session(
        sessionmaker(bind=_engine, expire_on_commit=False)
    )
    logger.info("Database initialised successfully.")


@contextmanager
def get_session() -> Generator:
    """Provide a transactional database session as a context manager.

    Commits automatically on clean exit from the ``with`` block.  On any
    exception, rolls back the transaction and re-raises the original error.
    The session is always closed at the end of the block.

    Must only be called after :func:`init_db` has been called.

    Yields:
        sqlalchemy.orm.Session: An active, thread-local database session.

    Raises:
        RuntimeError: If :func:`init_db` has not been called yet.

    Example::

        with get_session() as session:
            run = TrainingRun(name="exp-1", model_architecture="yolov8n")
            session.add(run)
        # Transaction committed automatically here.
    """
    if _SessionFactory is None:
        raise RuntimeError(
            "Database has not been initialised. Call init_db() before get_session()."
        )

    session = _SessionFactory()
    try:
        yield session
        session.commit()
    except Exception:
        session.rollback()
        raise
    finally:
        session.close()


def get_engine():
    """Return the active SQLAlchemy engine.

    Returns:
        The SQLAlchemy ``Engine`` instance, or ``None`` if :func:`init_db`
        has not been called yet.
    """
    return _engine
