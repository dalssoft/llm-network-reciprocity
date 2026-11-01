import base64
from enum import Enum


class MessageRole(Enum):
    SYSTEM = "system"
    USER = "user"
    ASSISTANT = "assistant"


class Message:
    def __init__(self, role: MessageRole, content: str):
        self.role = role
        self.content = content

    def __str__(self):
        return f"{self.content}"

    def to_dict(self):
        return {"role": self.role.value, "content": self.content}


class BlobAttachment(Message):
    def __init__(self, mime_type: str, payload: bytes):
        super().__init__(MessageRole.USER, "")
        self.mime_type = mime_type
        self.payload = payload
        self._base64_cache = None

    def to_base64(self):
        if self._base64_cache is None:
            self._base64_cache = base64.b64encode(self.payload).decode("utf-8")
        return self._base64_cache

    def to_dict(self):
        return {
            "role": self.role.value,
            "content": [
                {
                    "type": "image_url",
                    "image_url": {
                        "url": f"data:image/{self.mime_type};base64,{self.to_base64()}"
                    },
                },
            ],
        }
