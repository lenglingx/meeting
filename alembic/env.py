from logging.config import fileConfig
from alembic import context
from sqlalchemy import engine_from_config, pool
from app.config import settings
from app.core.database import Base
from app.models import import_models

if context.config.config_file_name:
    fileConfig(context.config.config_file_name)
import_models()
target_metadata = Base.metadata

def run_migrations_offline():
    context.configure(url=settings.MYSQL_DSN, target_metadata=target_metadata, literal_binds=True, dialect_opts={"paramstyle": "named"})
    with context.begin_transaction():
        context.run_migrations()

def run_migrations_online():
    configuration = context.config.get_section(context.config.config_ini_section) or {}
    configuration["sqlalchemy.url"] = settings.MYSQL_DSN
    connectable = engine_from_config(configuration, prefix="sqlalchemy.", poolclass=pool.NullPool)
    with connectable.connect() as connection:
        context.configure(connection=connection, target_metadata=target_metadata)
        with context.begin_transaction():
            context.run_migrations()

if context.is_offline_mode():
    run_migrations_offline()
else:
    run_migrations_online()
