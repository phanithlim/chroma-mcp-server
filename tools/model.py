from pydantic import BaseModel

class CollectionModel(BaseModel):
    name: str
    meta: dict
    
class DocumentModel(BaseModel):
    page_content: str
    metadata: dict