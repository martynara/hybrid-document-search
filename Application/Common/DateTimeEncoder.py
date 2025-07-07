import json
from datetime import datetime

class DateTimeEncoder(json.JSONEncoder):
    """
    Custom JSON encoder that handles datetime objects and objects with to_dict method.
    """
    def default(self, obj):
        if isinstance(obj, datetime):
            return obj.isoformat()
        # Handle any object that has a to_dict method (like Document, DocumentMetadata, DocumentChunk)
        if hasattr(obj, 'to_dict') and callable(getattr(obj, 'to_dict')):
            return obj.to_dict()
        return super().default(obj)