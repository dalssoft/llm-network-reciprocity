from enum import Enum
import inspect
from typing import Any, Callable, Dict
from src.infra.util.str_util import remove_indentation


class Action:
    def __init__(
        self,
        function: Callable,
    ):
        self.function = function
        self.name = function.__name__

    def description(self):
        return (
            f"({self.name}) {remove_indentation(self.function.__doc__.strip()).strip()}"
        )

    def full_description(self):
        return f"{self.description()}\n\nParameters:\n{self.parameters_description()}"

    def parameters_description(self):
        params = inspect.signature(self.function).parameters
        keys = list(params.keys())
        param_strs = []
        for key in keys:
            param = params[key]
            if hasattr(param.annotation, "__metadata__"):
                param_strs.append(
                    f"{param.name}: {param.annotation.__args__[0].__name__} - {','.join(param.annotation.__metadata__)}"
                )
            else:
                param_strs.append(f"{param}")
        return "\n".join(param_strs)


class ActionCall:
    def __init__(self, id: str, action: Action, arguments: Dict[str, Any]):
        self.id = id
        self.action = action
        self.arguments = arguments

    def call(self):
        return self.action.function(**self.arguments)


class ActionChoice(Enum):
    NONE_OR_ANY = "auto"  # None, one or more
    ANY = "required"  # One or more
    NONE = "none"  # None
