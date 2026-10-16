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
