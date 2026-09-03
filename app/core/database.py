"""Synchronous SQLAlchemy layer for MySQL/PyMySQL."""

import logging
from collections.abc import Generator

from sqlalchemy import create_engine, text
from sqlalchemy.orm import DeclarativeBase, Session, sessionmaker

from app.config import settings

logger = logging.getLogger("database")


class Base(DeclarativeBase):
    pass


engine = create_engine(
    settings.MYSQL_DSN,
    echo=settings.APP_DEBUG,
    pool_pre_ping=True,
    pool_recycle=settings.MYSQL_POOL_RECYCLE,
    pool_size=settings.MYSQL_POOL_SIZE,
    max_overflow=settings.MYSQL_MAX_OVERFLOW,
    future=True,
)
SessionLocal = sessionmaker(bind=engine, autoflush=False, expire_on_commit=False)


def get_db() -> Generator[Session, None, None]:
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()


def init_db() -> None:
    from app.models import import_models

    import_models()
    Base.metadata.create_all(bind=engine)
    logger.info("MySQL 表结构已同步")


def check_db_connection() -> bool:
    try:
        with engine.connect() as conn:
            conn.execute(text("SELECT 1"))
        return True
    except Exception as exc:  # noqa: BLE001
        logger.warning("MySQL 连接检查失败: %s", exc)
        return False
