from typing import Any


def hello_project(*args, **kwargs) -> Any | tuple[Any, ...]:
    print(args, kwargs)
    return  # type:ignore
