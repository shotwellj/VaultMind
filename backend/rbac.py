"""
Role-Based Access Control (RBAC) module for VaultMind.
SQLite-backed multi-user authentication and authorization system.
"""

import sqlite3
import hashlib
import json
import os
from datetime import datetime, timedelta
from enum import Enum
from typing import Optional, List, Dict, Tuple
from dataclasses import dataclass

try:
    import jwt
except ImportError:
    jwt = None

try:
    import bcrypt
except ImportError:
    bcrypt = None


class Role(Enum):
    """Role enumeration for VaultMind users."""
    ADMIN = "admin"
    PARTNER = "partner"
    ASSOCIATE = "associate"
    PARALEGAL = "paralegal"
    VIEWER = "viewer"


@dataclass
class User:
    """Represents a VaultMind user."""
    user_id: str
    username: str
    email: str
    role: Role
    created_at: str
    is_active: bool
    workspaces: List[str]


@dataclass
class Permission:
    """Represents a specific permission."""
    action: str
    resource: str
    role: Role


@dataclass
class AuditEntry:
    """Represents an audit log entry."""
    entry_id: str
    user_id: str
    action: str
    resource: str
    timestamp: str
    details: Dict


class RBACManager:
    """Manages role-based access control, authentication, and audit trails."""

    def __init__(self, db_path: Optional[str] = None):
        """Initialize RBAC manager with SQLite database."""
        if db_path is None:
            home = os.path.expanduser("~")
            db_path = os.path.join(home, ".vaultmind", "auth", "auth.db")

        os.makedirs(os.path.dirname(db_path), exist_ok=True)
        self.db_path = db_path
        self.jwt_secret = "vaultmind-secret-key"
        self._init_db()

    def _init_db(self) -> None:
        """Initialize database tables if they don't exist."""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()

        cursor.execute('''
            CREATE TABLE IF NOT EXISTS users (
                user_id TEXT PRIMARY KEY,
                username TEXT UNIQUE NOT NULL,
                email TEXT UNIQUE NOT NULL,
                password_hash TEXT NOT NULL,
                role TEXT NOT NULL,
                created_at TEXT NOT NULL,
                is_active BOOLEAN NOT NULL DEFAULT 1
            )
        ''')

        cursor.execute('''
            CREATE TABLE IF NOT EXISTS workspace_assignments (
                assignment_id TEXT PRIMARY KEY,
                user_id TEXT NOT NULL,
                workspace_id TEXT NOT NULL,
                assigned_at TEXT NOT NULL,
                FOREIGN KEY (user_id) REFERENCES users(user_id),
                UNIQUE(user_id, workspace_id)
            )
        ''')

        cursor.execute('''
            CREATE TABLE IF NOT EXISTS audit_log (
                entry_id TEXT PRIMARY KEY,
                user_id TEXT NOT NULL,
                action TEXT NOT NULL,
                resource TEXT NOT NULL,
                timestamp TEXT NOT NULL,
                details TEXT NOT NULL,
                FOREIGN KEY (user_id) REFERENCES users(user_id)
            )
        ''')

        cursor.execute('''
            CREATE TABLE IF NOT EXISTS tokens (
                token_hash TEXT PRIMARY KEY,
                user_id TEXT NOT NULL,
                issued_at TEXT NOT NULL,
                expires_at TEXT NOT NULL,
                FOREIGN KEY (user_id) REFERENCES users(user_id)
            )
        ''')

        conn.commit()
        conn.close()

    def _hash_password(self, password: str) -> str:
        """Hash password using bcrypt or sha256 fallback."""
        if bcrypt:
            return bcrypt.hashpw(password.encode(), bcrypt.gensalt()).decode()
        else:
            return hashlib.sha256(password.encode()).hexdigest()

    def _verify_password(self, password: str, password_hash: str) -> bool:
        """Verify password against hash."""
        if bcrypt:
            try:
                return bcrypt.checkpw(password.encode(), password_hash.encode())
            except (ValueError, TypeError):
                return False
        else:
            return hashlib.sha256(password.encode()).hexdigest() == password_hash

    def create_user(
        self,
        user_id: str,
        username: str,
        email: str,
        password: str,
        role: Role
    ) -> User:
        """Create a new user in the system."""
        password_hash = self._hash_password(password)
        created_at = datetime.utcnow().isoformat()

        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()

        try:
            cursor.execute('''
                INSERT INTO users
                (user_id, username, email, password_hash, role, created_at, is_active)
                VALUES (?, ?, ?, ?, ?, ?, ?)
            ''', (user_id, username, email, password_hash, role.value, created_at, True))
            conn.commit()
        finally:
            conn.close()

        return User(
            user_id=user_id,
            username=username,
            email=email,
            role=role,
            created_at=created_at,
            is_active=True,
            workspaces=[]
        )

    def authenticate(self, username: str, password: str) -> Optional[User]:
        """Authenticate user and return User object if valid."""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()

        cursor.execute(
            'SELECT user_id, username, email, role, created_at, is_active FROM users WHERE username = ?',
            (username,)
        )
        row = cursor.fetchone()
        conn.close()

        if not row:
            return None

        user_id, username_db, email, role_str, created_at, is_active = row

        if not is_active:
            return None

        cursor = sqlite3.connect(self.db_path).cursor()
        cursor.execute(
            'SELECT password_hash FROM users WHERE user_id = ?',
            (user_id,)
        )
        password_hash = cursor.fetchone()[0]
        sqlite3.connect(self.db_path).close()

        if not self._verify_password(password, password_hash):
            return None

        workspaces = self._get_user_workspaces(user_id)

        return User(
            user_id=user_id,
            username=username_db,
            email=email,
            role=Role(role_str),
            created_at=created_at,
            is_active=is_active,
            workspaces=workspaces
        )

    def get_user(self, user_id: str) -> Optional[User]:
        """Retrieve user by ID."""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()

        cursor.execute(
            'SELECT user_id, username, email, role, created_at, is_active FROM users WHERE user_id = ?',
            (user_id,)
        )
        row = cursor.fetchone()
        conn.close()

        if not row:
            return None

        user_id, username, email, role_str, created_at, is_active = row
        workspaces = self._get_user_workspaces(user_id)

        return User(
            user_id=user_id,
            username=username,
            email=email,
            role=Role(role_str),
            created_at=created_at,
            is_active=is_active,
            workspaces=workspaces
        )

    def update_role(self, user_id: str, new_role: Role) -> bool:
        """Update user role."""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()

        cursor.execute(
            'UPDATE users SET role = ? WHERE user_id = ?',
            (new_role.value, user_id)
        )
        conn.commit()
        conn.close()

        return cursor.rowcount > 0

    def delete_user(self, user_id: str) -> bool:
        """Delete a user account."""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()

        cursor.execute('DELETE FROM workspace_assignments WHERE user_id = ?', (user_id,))
        cursor.execute('DELETE FROM audit_log WHERE user_id = ?', (user_id,))
        cursor.execute('DELETE FROM tokens WHERE user_id = ?', (user_id,))
        cursor.execute('DELETE FROM users WHERE user_id = ?', (user_id,))

        conn.commit()
        conn.close()

        return cursor.rowcount > 0

    def generate_token(self, user_id: str, expires_in_hours: int = 24) -> str:
        """Generate a JWT token for a user."""
        if not jwt:
            raise RuntimeError("PyJWT not installed")

        now = datetime.utcnow()
        expires_at = now + timedelta(hours=expires_in_hours)

        payload = {
            'user_id': user_id,
            'iat': now,
            'exp': expires_at
        }

        token = jwt.encode(payload, self.jwt_secret, algorithm='HS256')

        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()

        token_hash = hashlib.sha256(token.encode()).hexdigest()
        cursor.execute('''
            INSERT INTO tokens (token_hash, user_id, issued_at, expires_at)
            VALUES (?, ?, ?, ?)
        ''', (token_hash, user_id, now.isoformat(), expires_at.isoformat()))

        conn.commit()
        conn.close()

        return token

    def validate_token(self, token: str) -> Optional[str]:
        """Validate JWT token and return user_id if valid."""
        if not jwt:
            raise RuntimeError("PyJWT not installed")

        try:
            payload = jwt.decode(token, self.jwt_secret, algorithms=['HS256'])
            user_id = payload.get('user_id')

            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()

            token_hash = hashlib.sha256(token.encode()).hexdigest()
            cursor.execute(
                'SELECT user_id FROM tokens WHERE token_hash = ? AND expires_at > ?',
                (token_hash, datetime.utcnow().isoformat())
            )
            result = cursor.fetchone()
            conn.close()

            return user_id if result else None
        except Exception:
            return None

    def can_read(self, user_id: str, resource_id: str) -> bool:
        """Check if user can read a resource."""
        user = self.get_user(user_id)
        if not user or not user.is_active:
            return False

        if user.role in [Role.ADMIN, Role.PARTNER]:
            return True

        return resource_id in user.workspaces

    def can_write(self, user_id: str, resource_id: str) -> bool:
        """Check if user can write to a resource."""
        user = self.get_user(user_id)
        if not user or not user.is_active:
            return False

        if user.role == Role.ADMIN:
            return True

        if user.role in [Role.PARTNER, Role.ASSOCIATE]:
            return resource_id in user.workspaces

        return False

    def can_admin(self, user_id: str) -> bool:
        """Check if user has admin permissions."""
        user = self.get_user(user_id)
        return user and user.is_active and user.role == Role.ADMIN

    def can_export(self, user_id: str, resource_id: str) -> bool:
        """Check if user can export a resource."""
        user = self.get_user(user_id)
        if not user or not user.is_active:
            return False

        if user.role == Role.ADMIN:
            return True

        if user.role in [Role.PARTNER, Role.ASSOCIATE]:
            return resource_id in user.workspaces

        return False

    def check_permission(
        self,
        user_id: str,
        action: str,
        resource_id: str
    ) -> bool:
        """Check if user has a specific permission."""
        if action == "read":
            return self.can_read(user_id, resource_id)
        elif action == "write":
            return self.can_write(user_id, resource_id)
        elif action == "admin":
            return self.can_admin(user_id)
        elif action == "export":
            return self.can_export(user_id, resource_id)

        return False

    def _get_user_workspaces(self, user_id: str) -> List[str]:
        """Get list of workspace IDs assigned to user."""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()

        cursor.execute(
            'SELECT workspace_id FROM workspace_assignments WHERE user_id = ?',
            (user_id,)
        )
        workspaces = [row[0] for row in cursor.fetchall()]
        conn.close()

        return workspaces

    def assign_workspace(self, user_id: str, workspace_id: str) -> bool:
        """Assign a workspace to a user."""
        import uuid
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()

        assignment_id = str(uuid.uuid4())
        assigned_at = datetime.utcnow().isoformat()

        try:
            cursor.execute('''
                INSERT INTO workspace_assignments
                (assignment_id, user_id, workspace_id, assigned_at)
                VALUES (?, ?, ?, ?)
            ''', (assignment_id, user_id, workspace_id, assigned_at))
            conn.commit()
            return True
        except sqlite3.IntegrityError:
            return False
        finally:
            conn.close()

    def get_audit_log(
        self,
        user_id: Optional[str] = None,
        limit: int = 100
    ) -> List[AuditEntry]:
        """Retrieve audit log entries."""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()

        if user_id:
            cursor.execute('''
                SELECT entry_id, user_id, action, resource, timestamp, details
                FROM audit_log
                WHERE user_id = ?
                ORDER BY timestamp DESC
                LIMIT ?
            ''', (user_id, limit))
        else:
            cursor.execute('''
                SELECT entry_id, user_id, action, resource, timestamp, details
                FROM audit_log
                ORDER BY timestamp DESC
                LIMIT ?
            ''', (limit,))

        rows = cursor.fetchall()
        conn.close()

        entries = []
        for row in rows:
            entry_id, user_id_val, action, resource, timestamp, details_str = row
            entries.append(AuditEntry(
                entry_id=entry_id,
                user_id=user_id_val,
                action=action,
                resource=resource,
                timestamp=timestamp,
                details=json.loads(details_str) if details_str else {}
            ))

        return entries

    def log_access(
        self,
        user_id: str,
        action: str,
        resource: str,
        details: Optional[Dict] = None
    ) -> None:
        """Log a user access event to the audit trail."""
        import uuid
        if details is None:
            details = {}

        entry_id = str(uuid.uuid4())
        timestamp = datetime.utcnow().isoformat()
        details_str = json.dumps(details)

        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()

        cursor.execute('''
            INSERT INTO audit_log
            (entry_id, user_id, action, resource, timestamp, details)
            VALUES (?, ?, ?, ?, ?, ?)
        ''', (entry_id, user_id, action, resource, timestamp, details_str))

        conn.commit()
        conn.close()


# ── Module-level convenience functions ────────────────────────
# These wrap the RBACManager class for easy importing

_manager = None

def _get_manager():
    global _manager
    if _manager is None:
        _manager = RBACManager()
    return _manager

def create_user(username, password, email="", role="viewer"):
    return _get_manager().create_user(username, password, email, Role(role) if isinstance(role, str) else role)

def authenticate(username, password):
    user = _get_manager().authenticate(username, password)
    if user:
        return {"id": user.user_id, "username": user.username, "email": user.email, "role": user.role.value}
    return None

def generate_token(user_id, username="", role="viewer"):
    return _get_manager().generate_token(user_id)

def validate_token(token):
    return _get_manager().validate_token(token)

def check_permission(role, action):
    return _get_manager().check_permission("system", action)

def get_audit_log(limit=100, user_id=None):
    entries = _get_manager().get_audit_log(limit=limit, user_id=user_id)
    return [{"entry_id": e.entry_id, "user_id": e.user_id, "action": e.action,
             "resource": e.resource, "timestamp": e.timestamp} for e in entries]

def assign_workspace(user_id, workspace):
    return _get_manager().assign_workspace(user_id, workspace)

def get_user(user_id):
    user = _get_manager().get_user(user_id)
    if user:
        return {"id": user.user_id, "username": user.username, "email": user.email, "role": user.role.value}
    return None

def list_users():
    mgr = _get_manager()
    conn = sqlite3.connect(mgr.db_path)
    rows = conn.execute("SELECT user_id, username, email, role FROM users").fetchall()
    conn.close()
    return [{"id": r[0], "username": r[1], "email": r[2], "role": r[3]} for r in rows]

def update_role(user_id, role):
    return _get_manager().update_role(user_id, Role(role) if isinstance(role, str) else role)
