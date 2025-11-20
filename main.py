import os
from datetime import datetime, timedelta, timezone
from typing import List, Optional, Dict, Any

from fastapi import FastAPI, HTTPException, Depends, Header, WebSocket, WebSocketDisconnect
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, EmailStr
from bson import ObjectId

from database import db, create_document, get_documents

app = FastAPI(title="Project Management API")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# -----------------------------
# Helpers
# -----------------------------

def oid(value: str) -> ObjectId:
    try:
        return ObjectId(value)
    except Exception:
        raise HTTPException(status_code=400, detail="Invalid id")


def serialize(doc: Dict[str, Any]) -> Dict[str, Any]:
    if not doc:
        return doc
    d = {**doc}
    if "_id" in d:
        d["id"] = str(d.pop("_id"))
    for k, v in list(d.items()):
        if isinstance(v, datetime):
            d[k] = v.astimezone(timezone.utc).isoformat()
        if isinstance(v, ObjectId):
            d[k] = str(v)
    return d


def hash_password(pw: str) -> str:
    import hashlib
    return hashlib.sha256(pw.encode()).hexdigest()


# -----------------------------
# Schemas (subset for requests)
# -----------------------------
class RegisterRequest(BaseModel):
    name: str
    email: EmailStr
    password: str


class LoginRequest(BaseModel):
    email: EmailStr
    password: str


class ProjectCreate(BaseModel):
    name: str
    description: Optional[str] = None


class TaskCreate(BaseModel):
    title: str
    description: Optional[str] = None
    assignee_id: Optional[str] = None
    due_date: Optional[str] = None


class TaskUpdate(BaseModel):
    title: Optional[str] = None
    description: Optional[str] = None
    status: Optional[str] = None
    assignee_id: Optional[str] = None
    due_date: Optional[str] = None


class CommentCreate(BaseModel):
    content: str


# -----------------------------
# Auth utilities
# -----------------------------
async def get_current_user(authorization: Optional[str] = Header(default=None)) -> Dict[str, Any]:
    if not authorization or not authorization.lower().startswith("bearer "):
        raise HTTPException(status_code=401, detail="Missing token")
    token = authorization.split(" ", 1)[1]
    session = db["session"].find_one({"token": token})
    if not session:
        raise HTTPException(status_code=401, detail="Invalid token")
    if session.get("expires_at") and session["expires_at"] < datetime.now(timezone.utc):
        raise HTTPException(status_code=401, detail="Session expired")
    user = db["user"].find_one({"_id": session["user_id"]})
    if not user:
        raise HTTPException(status_code=401, detail="User not found")
    return serialize(user)


# -----------------------------
# Auth endpoints
# -----------------------------
@app.post("/auth/register")
def register(body: RegisterRequest):
    existing = db["user"].find_one({"email": body.email})
    if existing:
        raise HTTPException(status_code=400, detail="Email already registered")
    payload = {
        "name": body.name,
        "email": body.email,
        "password": hash_password(body.password),
        "created_at": datetime.now(timezone.utc),
        "updated_at": datetime.now(timezone.utc),
    }
    res = db["user"].insert_one(payload)
    user_id = str(res.inserted_id)
    return {"id": user_id, "name": body.name, "email": body.email}


@app.post("/auth/login")
def login(body: LoginRequest):
    user = db["user"].find_one({"email": body.email})
    if not user or user.get("password") != hash_password(body.password):
        raise HTTPException(status_code=401, detail="Invalid credentials")
    import secrets
    token = secrets.token_urlsafe(32)
    session = {
        "token": token,
        "user_id": user["_id"],
        "created_at": datetime.now(timezone.utc),
        "expires_at": datetime.now(timezone.utc) + timedelta(days=7),
    }
    db["session"].insert_one(session)
    return {"token": token, "user": serialize(user)}


@app.get("/me")
async def me(user=Depends(get_current_user)):
    return user


# -----------------------------
# Project endpoints
# -----------------------------
@app.post("/projects")
async def create_project(body: ProjectCreate, user=Depends(get_current_user)):
    doc = {
        "name": body.name,
        "description": body.description,
        "owner_id": ObjectId(user["id"]),
        "member_ids": [ObjectId(user["id"])],
        "created_at": datetime.now(timezone.utc),
        "updated_at": datetime.now(timezone.utc),
    }
    res = db["project"].insert_one(doc)
    project = db["project"].find_one({"_id": res.inserted_id})
    return serialize(project)


@app.get("/projects")
async def list_projects(user=Depends(get_current_user)):
    cursor = db["project"].find({"member_ids": ObjectId(user["id"])})
    return [serialize(p) for p in cursor]


@app.get("/projects/{project_id}")
async def get_project(project_id: str, user=Depends(get_current_user)):
    p = db["project"].find_one({"_id": oid(project_id), "member_ids": ObjectId(user["id"])})
    if not p:
        raise HTTPException(status_code=404, detail="Project not found")
    return serialize(p)


class AddMemberRequest(BaseModel):
    user_id: str


@app.post("/projects/{project_id}/members")
async def add_member(project_id: str, body: AddMemberRequest, user=Depends(get_current_user)):
    p = db["project"].find_one({"_id": oid(project_id)})
    if not p:
        raise HTTPException(status_code=404, detail="Project not found")
    if str(p.get("owner_id")) != user["id"]:
        raise HTTPException(status_code=403, detail="Only owner can add members")
    db["project"].update_one(
        {"_id": p["_id"]},
        {"$addToSet": {"member_ids": oid(body.user_id)}, "$set": {"updated_at": datetime.now(timezone.utc)}}
    )
    updated = db["project"].find_one({"_id": p["_id"]})
    return serialize(updated)


# -----------------------------
# Task endpoints
# -----------------------------
@app.post("/projects/{project_id}/tasks")
async def create_task(project_id: str, body: TaskCreate, user=Depends(get_current_user)):
    p = db["project"].find_one({"_id": oid(project_id), "member_ids": ObjectId(user["id"])})
    if not p:
        raise HTTPException(status_code=404, detail="Project not found or access denied")
    doc = {
        "project_id": p["_id"],
        "title": body.title,
        "description": body.description,
        "status": "todo",
        "assignee_id": oid(body.assignee_id) if body.assignee_id else None,
        "due_date": body.due_date,
        "created_by": ObjectId(user["id"]),
        "created_at": datetime.now(timezone.utc),
        "updated_at": datetime.now(timezone.utc),
    }
    res = db["task"].insert_one(doc)
    task = serialize(db["task"].find_one({"_id": res.inserted_id}))
    await broadcast_project(project_id, {"type": "task_created", "task": task})
    # Notification for assignee
    if body.assignee_id and body.assignee_id != user["id"]:
        create_notification(body.assignee_id, "task_assigned", f"You were assigned to '{body.title}'")
    return task


@app.get("/projects/{project_id}/tasks")
async def list_tasks(project_id: str, user=Depends(get_current_user)):
    p = db["project"].find_one({"_id": oid(project_id), "member_ids": ObjectId(user["id"])})
    if not p:
        raise HTTPException(status_code=404, detail="Project not found or access denied")
    cursor = db["task"].find({"project_id": p["_id"]}).sort("created_at", 1)
    return [serialize(t) for t in cursor]


@app.patch("/tasks/{task_id}")
async def update_task(task_id: str, body: TaskUpdate, user=Depends(get_current_user)):
    task = db["task"].find_one({"_id": oid(task_id)})
    if not task:
        raise HTTPException(status_code=404, detail="Task not found")
    project = db["project"].find_one({"_id": task["project_id"], "member_ids": ObjectId(user["id"])})
    if not project:
        raise HTTPException(status_code=403, detail="Access denied")
    update: Dict[str, Any] = {}
    for field in ["title", "description", "status", "due_date"]:
        val = getattr(body, field)
        if val is not None:
            update[field] = val
    if body.assignee_id is not None:
        update["assignee_id"] = oid(body.assignee_id) if body.assignee_id else None
    if not update:
        return serialize(task)
    update["updated_at"] = datetime.now(timezone.utc)
    db["task"].update_one({"_id": task["_id"]}, {"$set": update})
    updated = serialize(db["task"].find_one({"_id": task["_id"]}))
    await broadcast_project(str(project["_id"]), {"type": "task_updated", "task": updated})
    # Notify assignee on status changes
    if "assignee_id" in update and update["assignee_id"] and str(update["assignee_id"]) != user["id"]:
        create_notification(str(update["assignee_id"]), "task_updated", f"Task '{updated['title']}' updated")
    return updated


# -----------------------------
# Comment endpoints
# -----------------------------
@app.get("/tasks/{task_id}/comments")
async def get_comments(task_id: str, user=Depends(get_current_user)):
    task = db["task"].find_one({"_id": oid(task_id)})
    if not task:
        raise HTTPException(status_code=404, detail="Task not found")
    project = db["project"].find_one({"_id": task["project_id"], "member_ids": ObjectId(user["id"])})
    if not project:
        raise HTTPException(status_code=403, detail="Access denied")
    cursor = db["comment"].find({"task_id": task["_id"]}).sort("created_at", 1)
    return [serialize(c) for c in cursor]


@app.post("/tasks/{task_id}/comments")
async def add_comment(task_id: str, body: CommentCreate, user=Depends(get_current_user)):
    task = db["task"].find_one({"_id": oid(task_id)})
    if not task:
        raise HTTPException(status_code=404, detail="Task not found")
    project = db["project"].find_one({"_id": task["project_id"], "member_ids": ObjectId(user["id"])})
    if not project:
        raise HTTPException(status_code=403, detail="Access denied")
    doc = {
        "task_id": task["_id"],
        "author_id": ObjectId(user["id"]),
        "content": body.content,
        "created_at": datetime.now(timezone.utc),
        "updated_at": datetime.now(timezone.utc),
    }
    res = db["comment"].insert_one(doc)
    comment = serialize(db["comment"].find_one({"_id": res.inserted_id}))
    await broadcast_project(str(project["_id"]), {"type": "comment_added", "task_id": str(task["_id"]), "comment": comment})
    # notify assignee if not author
    if task.get("assignee_id") and str(task["assignee_id"]) != user["id"]:
        create_notification(str(task["assignee_id"]), "comment_added", f"New comment on '{task['title']}'")
    return comment


# -----------------------------
# Notifications
# -----------------------------
@app.get("/notifications")
async def list_notifications(user=Depends(get_current_user)):
    cursor = db["notification"].find({"user_id": ObjectId(user["id"])}, sort=[("created_at", -1)]).limit(50)
    notes = [serialize(n) for n in cursor]
    return notes


class MarkReadRequest(BaseModel):
    notification_ids: List[str]


@app.post("/notifications/read")
async def mark_notifications_read(body: MarkReadRequest, user=Depends(get_current_user)):
    ids = [oid(i) for i in body.notification_ids]
    db["notification"].update_many({"_id": {"$in": ids}, "user_id": ObjectId(user["id"])}, {"$set": {"read": True}})
    return {"updated": len(ids)}


def create_notification(user_id: str, type_: str, message: str):
    db["notification"].insert_one({
        "user_id": ObjectId(user_id),
        "type": type_,
        "message": message,
        "read": False,
        "created_at": datetime.now(timezone.utc),
        "updated_at": datetime.now(timezone.utc),
    })


# -----------------------------
# WebSocket manager per project
# -----------------------------
class ConnectionManager:
    def __init__(self):
        self.project_connections: Dict[str, List[WebSocket]] = {}

    async def connect(self, project_id: str, websocket: WebSocket):
        await websocket.accept()
        self.project_connections.setdefault(project_id, []).append(websocket)

    def disconnect(self, project_id: str, websocket: WebSocket):
        conns = self.project_connections.get(project_id, [])
        if websocket in conns:
            conns.remove(websocket)
        if not conns and project_id in self.project_connections:
            del self.project_connections[project_id]

    async def broadcast(self, project_id: str, message: Dict[str, Any]):
        for ws in list(self.project_connections.get(project_id, [])):
            try:
                await ws.send_json(message)
            except Exception:
                self.disconnect(project_id, ws)


manager = ConnectionManager()


async def broadcast_project(project_id: str, message: Dict[str, Any]):
    await manager.broadcast(project_id, message)


@app.websocket("/ws/projects/{project_id}")
async def project_ws(websocket: WebSocket, project_id: str):
    await manager.connect(project_id, websocket)
    try:
        while True:
            # Keep alive / receive pings from client if any
            await websocket.receive_text()
    except WebSocketDisconnect:
        manager.disconnect(project_id, websocket)


# -----------------------------
# Health/Test
# -----------------------------
@app.get("/")
def read_root():
    return {"message": "Project Management API running"}


@app.get("/test")
def test_database():
    response = {
        "backend": "✅ Running",
        "database": "❌ Not Available",
        "database_url": None,
        "database_name": None,
        "connection_status": "Not Connected",
        "collections": []
    }
    try:
        if db is not None:
            response["database"] = "✅ Available"
            response["database_url"] = "✅ Set" if os.getenv("DATABASE_URL") else "❌ Not Set"
            response["database_name"] = "✅ Set" if os.getenv("DATABASE_NAME") else "❌ Not Set"
            response["connection_status"] = "Connected"
            try:
                collections = db.list_collection_names()
                response["collections"] = collections[:10]
                response["database"] = "✅ Connected & Working"
            except Exception as e:
                response["database"] = f"⚠️ Connected but Error: {str(e)[:50]}"
    except Exception as e:
        response["database"] = f"❌ Error: {str(e)[:50]}"
    return response


if __name__ == "__main__":
    import uvicorn
    port = int(os.getenv("PORT", 8000))
    uvicorn.run(app, host="0.0.0.0", port=port)
