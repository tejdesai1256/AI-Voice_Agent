from datetime import datetime, timezone
from typing import Optional
from fastapi import APIRouter, Depends, HTTPException, status
from pydantic import BaseModel
from sqlalchemy.orm import Session

from app.database.database import get_db
from app.database.models import SystemUser
from app.utils.auth import (
    hash_password,
    verify_password,
    create_access_token,
    get_current_user,
    require_admin,
)

router = APIRouter()


class LoginRequest(BaseModel):
    username: str
    password: str


class CreateUserRequest(BaseModel):
    username: str
    password: str
    role: str = "system_user"


class UserResponse(BaseModel):
    id: int
    username: str
    role: str
    created_at: Optional[str] = None


@router.post("/login")
def login(req: LoginRequest, db: Session = Depends(get_db)):
    """Authenticate system user and return JWT access token."""
    username = req.username.strip()
    user = db.query(SystemUser).filter(SystemUser.username == username).first()

    if not user or not verify_password(req.password, user.password_hash):
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid username or password",
        )

    token = create_access_token(data={"sub": user.username, "role": user.role, "uid": user.id})

    return {
        "access_token": token,
        "token_type": "bearer",
        "user": {
            "id": user.id,
            "username": user.username,
            "role": user.role,
        },
    }


@router.post("/logout")
def logout(current_user: SystemUser = Depends(get_current_user)):
    """Logs out current user session."""
    return {"status": "logged_out", "username": current_user.username}


@router.get("/me")
def get_me(current_user: SystemUser = Depends(get_current_user)):
    """Returns profile information for the authenticated user."""
    return {
        "id": current_user.id,
        "username": current_user.username,
        "role": current_user.role,
        "created_at": current_user.created_at.isoformat() if current_user.created_at else None,
    }


@router.get("/users")
def list_system_users(
    admin: SystemUser = Depends(require_admin),
    db: Session = Depends(get_db),
):
    """Admin-only endpoint to list all registered system users."""
    users = db.query(SystemUser).order_by(SystemUser.id.asc()).all()
    return [
        {
            "id": u.id,
            "username": u.username,
            "role": u.role,
            "created_at": u.created_at.isoformat() if u.created_at else None,
            "updated_at": u.updated_at.isoformat() if u.updated_at else None,
        }
        for u in users
    ]


@router.post("/users")
def create_system_user(
    req: CreateUserRequest,
    admin: SystemUser = Depends(require_admin),
    db: Session = Depends(get_db),
):
    """Admin-only endpoint to create a new system user."""
    username = req.username.strip()
    if not username:
        raise HTTPException(status_code=400, detail="Username is required")
    if not req.password or len(req.password) < 6:
        raise HTTPException(status_code=400, detail="Password must be at least 6 characters")
    if req.role not in ["admin", "system_user"]:
        raise HTTPException(status_code=400, detail="Role must be 'admin' or 'system_user'")

    existing = db.query(SystemUser).filter(SystemUser.username == username).first()
    if existing:
        raise HTTPException(status_code=400, detail=f"User '{username}' already exists")

    now = datetime.now(timezone.utc)
    new_user = SystemUser(
        username=username,
        password_hash=hash_password(req.password),
        role=req.role,
        created_at=now,
        updated_at=now,
    )
    db.add(new_user)
    db.commit()
    db.refresh(new_user)

    return {
        "id": new_user.id,
        "username": new_user.username,
        "role": new_user.role,
        "created_at": new_user.created_at.isoformat(),
    }


@router.delete("/users/{user_id}")
def delete_system_user(
    user_id: int,
    admin: SystemUser = Depends(require_admin),
    db: Session = Depends(get_db),
):
    """Admin-only endpoint to delete a system user."""
    if user_id == admin.id:
        raise HTTPException(status_code=400, detail="Cannot delete your own admin account")

    target = db.query(SystemUser).filter(SystemUser.id == user_id).first()
    if not target:
        raise HTTPException(status_code=404, detail="User not found")

    db.delete(target)
    db.commit()
    return {"status": "deleted", "user_id": user_id}
