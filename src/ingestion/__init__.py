"""DataPilot AI ingestion package."""

from src.ingestion.collector import DocumentCollector
from src.ingestion.inventory import list_topic_targets
from src.ingestion.models import CollectionRecord, TopicTarget

__all__ = [
    "CollectionRecord",
    "DocumentCollector",
    "TopicTarget",
    "list_topic_targets",
]
