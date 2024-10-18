from collections import namedtuple
from typing import Any, Callable, Iterable, NamedTuple, Optional

from jinja2 import Template

from hello import hello_project
from kubeflow_from_pipeline.databases import gs_read_auto, gs_store_auto, load_gs_data

GLOBAL_IMPORTS = """
from typing import Any, NamedTuple
import kfp.dsl as dsl
from collections import namedtuple
from kfp.dsl import Input, Output, Dataset, component
from consts import MLFLOW_IMAGE
"""

DECLARATIVE_FUNCTION_TEMPLATE = Template(
    """
{{ decorator }}
def {{ function_name }}({{ parameters|join(', ') }}) -> {{ output_type_hint }}:
    # Imports
    {{ imports|join('\n')|indent(4) }}
    # Body
    {{ body|indent(4) }}
"""
)


def make_input(params: Iterable) -> list[str]:
    return list(map(lambda x: f"{x}: Input[Dataset]", params))


def make_output(params: Iterable) -> list[str]:
    return list(map(lambda x: f"{x}: Output[Dataset]", params))


def create_function_code(
    func_name: str,
    params: list[str],
    body: str,
    imports: list[str],
    output_type_hint: str = "Any",
    decorator: Optional[str] = None,
) -> str:

    if decorator is None:
        decorator = ""

    code = DECLARATIVE_FUNCTION_TEMPLATE.render(
        function_name=func_name,
        parameters=params,
        output_type_hint=output_type_hint,
        imports=imports,
        body=body,
        decorator=decorator,
    )

    return code


def create_import_statement(function: Callable) -> str:
    return f"from {function.__module__} import {function.__name__}"


def create_function_call(
    function_name: str, args: list[str], kwargs: dict[str, Any], outputs: list[str]
) -> str:

    function_arguments = args + [f"{k}={repr(v)}" for k, v in kwargs.items()]

    if outputs:
        function_outputs = ", ".join(outputs) + " = "
    else:
        function_outputs = ""

    return f"{function_outputs}{function_name}({', '.join(function_arguments)})"


def create_function_from_step(
    name,
    function: Callable,
    inputs: Optional[list[str]] = None,
    outputs: Optional[list[str]] = None,
    kwargs: Optional[dict] = None,
    decorator: Optional[str] = None,
) -> str:

    if inputs is None:
        inputs = []

    if outputs is None:
        outputs = []

    if kwargs is None:
        kwargs = {}

    component_name = "_".join(([function.__name__, name] + outputs))

    imports = []

    if inputs:
        imports.append(create_import_statement(gs_read_auto))

    if outputs:
        imports.append(create_import_statement(gs_store_auto))

    imports.append(create_import_statement(function))

    input_params = list(map(lambda x: f"{x}_dataset", inputs))
    loading_lines = [
        f"{inpt} = gs_read_auto({param}.path)"
        for param, inpt in zip(input_params, inputs)
    ]

    loading_statement = "\n".join(loading_lines)

    function_call = create_function_call(
        function_name=function.__name__, args=inputs, kwargs=kwargs, outputs=outputs
    )

    output_params = list(map(lambda x: f"{x}_dataset", outputs))

    output_lines = [
        f"gs_store_auto({outpt}, {param}.path)"
        for outpt, param in zip(outputs, output_params)
    ]

    output_statement = "\n".join(output_lines)

    body = "\n".join(
        [
            "# Inputs Loading",
            loading_statement,
            "# Function Call",
            function_call,
            "# Outputs Packing",
            output_statement,
        ]
    )

    output_type_hint = "None"

    component_params = make_input(input_params) + make_output(output_params)

    component_code = create_function_code(
        component_name,
        component_params,
        body,
        imports,
        output_type_hint,
        decorator=decorator,
    )

    return component_code


if __name__ == "__main__":

    step = {
        "name": "raw",
        "function": hello_project,
        "inputs": ["a", "b"],
        "outputs": ["catalog"],
        "kwargs": {"sql_file": "data.sql", "n_iqr": 1.5},
    }

    func_name = "multiply_numbers"
    params = [
        "a",
        "b",
    ]
    body = """

    catalog_full = combine_catalog_sources(a, b)

    outputs = namedtuple("outputs", catalog_full=dsl.Dataset)
    return outputs(catalog_full=catalog_full)"""
    imports = [
        "from src.databases.load_gs_data import gs_read_df",
        "from src.databases.data_collection import combine_catalog_sources",
    ]
    output_type_hint = 'NamedTuple("outputs", catalog_full=dsl.Dataset)'

    decorator = "@component(base_image=MLFLOW_IMAGE)"

    code = GLOBAL_IMPORTS
    code += create_function_code(func_name, params, body, imports, output_type_hint)
    code += create_function_code(
        func_name, make_input(params), body, imports, decorator=decorator
    )
    name = step["name"]
    function: Callable = step["function"]
    inputs: list = step["inputs"]
    outputs: list = step["outputs"]
    kwargs: dict = step["kwargs"]

    code += create_function_from_step(
        name,
        function,
        inputs,
        outputs,
        kwargs,
        decorator="@component(base_image=MLFLOW_IMAGE)",
    )

    with open("generated_functions.py", "w") as f:
        f.write(code)
