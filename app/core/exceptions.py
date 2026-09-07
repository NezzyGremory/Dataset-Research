class DatasetResearchError(Exception):
    """Base exception for Dataset Research."""


class DatasetLoadError(DatasetResearchError):
    """Raised when a dataset cannot be loaded."""