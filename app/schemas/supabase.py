from pydantic import BaseModel


class SupabaseHealthResponse(BaseModel):
    connected: bool
    status_code: int | None = None
    message: str

