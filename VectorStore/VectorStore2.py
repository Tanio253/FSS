from llama_index.core.bridge.pydantic import Field
from VectorStore.BaseVectorStore import BaseVectorStore
from llama_index.core.schema import BaseNode
from typing import List, Any, Dict

class VectorStore2(BaseVectorStore):
    """VectorStore2 (add/get/delete implemented)."""

    stores_text: bool = True
    node_dict: Dict[str, BaseNode] = Field(default_factory=dict)

    def get(self, text_id: str) -> List[float]:
        return self.node_dict[text_id]

    def add(
        self,
        nodes: List[BaseNode],
    ) -> List[str]:
        for node in nodes:
            self.node_dict[node.node_id] = node

    def delete(self, node_id: str, **delete_kwargs: Any) -> None:
        del self.node_dict[node_id]