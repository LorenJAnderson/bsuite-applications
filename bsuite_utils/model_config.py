import dataclasses


@dataclasses.dataclass
class ModelConfig:
    name: str
    cls: type
    policy: str = "MlpPolicy"
    kwargs: dict = dataclasses.field(default_factory=dict)

