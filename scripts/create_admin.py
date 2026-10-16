import argparse
import sys
import os
from datetime import datetime, timezone

# Ensure project root is on sys.path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from app.database.database import SessionLocal, Base, engine
import app.database.models  # noqa: F401
from app.database.models import SystemUser
from app.utils.auth import hash_password


def create_or_update_admin(username: str, password: str, role: str = "admin"):
    if not username or not username.strip():
        print("Error: Username cannot be empty.")
        sys.exit(1)
    if not password or len(password) < 6:
        print("Error: Password must be at least 6 characters.")
        sys.exit(1)

    # Ensure all tables exist in the database
    Base.metadata.create_all(bind=engine)

    username = username.strip()
    hashed = hash_password(password)

    with SessionLocal() as db:

        user = db.query(SystemUser).filter(SystemUser.username == username).first()
        now = datetime.now(timezone.utc)
        if user:
            user.password_hash = hashed
            user.role = role
            user.updated_at = now
            db.commit()
            print(f"Successfully updated user '{username}' with role '{role}'.")
        else:
            new_user = SystemUser(
                username=username,
                password_hash=hashed,
                role=role,
                created_at=now,
                updated_at=now,
            )
            db.add(new_user)
            db.commit()
            db.refresh(new_user)
            print(f"Successfully created new {role} user '{username}' (ID: {new_user.id}).")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Create or update a system admin user.")
    parser.add_argument("--username", "-u", type=str, help="Admin username")
    parser.add_argument("--password", "-p", type=str, help="Admin password")
    parser.add_argument("--role", "-r", type=str, default="admin", choices=["admin", "system_user"], help="User role")

    args = parser.parse_args()

    username = args.username
    password = args.password
    role = args.role

    if not username:
        username = input("Enter admin username: ").strip()
    if not password:
        import getpass
        password = getpass.getpass("Enter admin password: ").strip()

    create_or_update_admin(username, password, role)
