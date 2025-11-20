"""
Database Schemas for Project Management Tool

Each Pydantic model represents a MongoDB collection. The collection name is
lowercased from the class name, e.g. User -> "user".
"""
from typing import List, Optional, Literal
from pydantic import BaseModel, Field, EmailStr

# Auth and Users
class User(BaseModel):
    name: str = Field(..., description="Full name")
    email: EmailStr = Field(..., description="Unique email")
    password: str = Field(..., description="Plain password on create; not returned in queries")

class UserPublic(BaseModel):
    id: str
    name: str
    email: EmailStr

# Projects
class Project(BaseModel):
    name: str
    description: Optional[str] = None
    owner_id: str = Field(..., description="User id of project owner")
    member_ids: List[str] = Field(default_factory=list, description="User ids that can access the project")

# Tasks
TaskStatus = Literal["todo", "in_progress", "done"]
class Task(BaseModel):
    project_id: str
    title: str
    description: Optional[str] = None
    status: TaskStatus = "todo"
    assignee_id: Optional[str] = None
    due_date: Optional[str] = Field(None, description="ISO date string")

# Comments
class Comment(BaseModel):
    task_id: str
    author_id: str
    content: str

# Notifications
class Notification(BaseModel):
    user_id: str
    type: Literal["task_assigned", "task_updated", "comment_added", "project_invite"]
    message: str
    read: bool = False
