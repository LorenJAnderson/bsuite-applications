import dataclasses


@dataclasses.dataclass
class ModelConfig:
    name: str
    cls: type
    kwargs: dict
