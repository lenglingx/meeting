"""Initial meeting schema."""
from alembic import op
from app.core.database import Base
from app.models import import_models

revision = "0001_initial"
down_revision = None
branch_labels = None
depends_on = None

def upgrade():
    import_models()
    Base.metadata.create_all(bind=op.get_bind())

def downgrade():
    import_models()
    Base.metadata.drop_all(bind=op.get_bind())
