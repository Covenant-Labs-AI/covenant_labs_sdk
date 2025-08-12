from dataclasses import dataclass
from enum import Enum
from typing import Optional, List


class DeploymentStatus(Enum):
    ENCRYPTION_PENDING = "ENCRYPTION_PENDING"
    DEPLOYING = "DEPLOYING"
    DEPLOYED = "DEPLOYED"


@dataclass
class SecureInferenceRequest:
    tokens: List[int]


@dataclass
class SecureInferenceResponse:
    generated_tokens: List[int]


@dataclass
class Model:
    id: str
    name: str
    vocab_size: int
    num_embeddings: int
    embedding_column_size: int
    hidden_size: int
    description: str
    hidden_layers: int
    attention_heads: int
    kv_heads: int
    model_params: int
    vision_hidden_size: Optional[int]  # if multimodal
    max_context_length: int
    hugging_face_url: str
    encrypted_layers: List[str]  # layers of the model that must be permuted


@dataclass
class Deployment:
    id: str
    user_id: str
    model: Model
    status: str


@dataclass
class ApiErrorResponse:
    error: str
