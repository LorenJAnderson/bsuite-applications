import dataclasses


@dataclasses.dataclass
class ModelConfig:
    name: str
    cls: type
    kwargs: dict = dataclasses.field(default_factory=dict)

