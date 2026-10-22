import os
from dotenv import load_dotenv
from sqlalchemy import create_engine
from sqlalchemy.orm import declarative_base, sessionmaker

# Load environment variables from .env file
load_dotenv()

# Read DATABASE_URL from environment variable
DATABASE_URL = os.getenv("DATABASE_URL")

if not DATABASE_URL:
    DATABASE_URL = "sqlite:///./voice_agent.db"
    print(
        "[Database Notice] DATABASE_URL not set in .env. "
        "Using local SQLite database: sqlite:///./voice_agent.db\n"
        "To use MySQL, set DATABASE_URL=mysql+pymysql://<user>:<password>@localhost:3306/<dbname> in .env"
    )

is_sqlite = DATABASE_URL.startswith("sqlite")
connect_args = {"check_same_thread": False} if is_sqlite else {}

# Create SQLAlchemy engine
engine = create_engine(
    DATABASE_URL,
    pool_pre_ping=True,
    echo=False,
    connect_args=connect_args,
)


# Create SessionLocal session factory
SessionLocal = sessionmaker(
    autocommit=False,
    autoflush=False,
    bind=engine,
)

# Base class for declarative ORM models
Base = declarative_base()


def ensure_schema_migrations(target_engine=None):
    """
    Safely inspects existing tables and adds any newly introduced columns
    to maintain backward compatibility with existing databases without requiring
    manual SQL scripts.
    """
    engine_to_use = target_engine or engine
    from sqlalchemy import inspect, text

    try:
        inspector = inspect(engine_to_use)
        if "calls" in inspector.get_table_names():
            columns = [c["name"] for c in inspector.get_columns("calls")]
            with engine_to_use.begin() as conn:
                if "provider" not in columns:
                    conn.execute(text("ALTER TABLE calls ADD COLUMN provider VARCHAR(50) DEFAULT 'web'"))
                if "provider_call_id" not in columns:
                    conn.execute(text("ALTER TABLE calls ADD COLUMN provider_call_id VARCHAR(255)"))
                if "caller_phone" not in columns:
                    conn.execute(text("ALTER TABLE calls ADD COLUMN caller_phone VARCHAR(50)"))
                if "destination_phone" not in columns:
                    conn.execute(text("ALTER TABLE calls ADD COLUMN destination_phone VARCHAR(50)"))
                if "duration_seconds" not in columns:
                    conn.execute(text("ALTER TABLE calls ADD COLUMN duration_seconds INTEGER"))
                if "error_code" not in columns:
                    conn.execute(text("ALTER TABLE calls ADD COLUMN error_code VARCHAR(100)"))
    except Exception as e:
        print(f"[DB Notice] Schema check: {e}")


def get_db():
    """
    FastAPI dependency that yields a database session for a request
    and ensures clean closure in a finally block.
    """
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()

