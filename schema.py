from pydantic import BaseModel
from typing import Optional

class RequestSchema(BaseModel):
    query: str
    role: Optional[str] = None
    exp_in_years: Optional[float] = None
    company: Optional[str] = None