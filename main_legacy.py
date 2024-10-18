from collections import namedtuple
from typing import Any, Callable, Iterable, NamedTuple, Optional

from jinja2 import Template

from hello import hello_project
from kubeflow_from_pipeline.databases import gs_read_auto, load_gs_data

GLOBAL_IMPORTS = """
from typing import Any, NamedTuple
import kfp.dsl as dsl
from collections import namedtuple
from kfp.dsl import Input, Output, Dataset, component
from consts import MLFLOW_IMAGE
"""


TUPLE_FUNCTION_TEMPLATE = Template(
    """
{{ decorator }}
def {{ function_name }}({{ parameters|join(', ') }}) -> {{ output_type_hint }}:
    # Imports
    {{ imports|join('\n')|indent(4) }}
    # Body
    {{ body|indent(4) }}

"""
)

DECLARATIVE_FUNCTION_TEMPLATE = Template(
    """
{{ decorator }}
def {{ function_name }}({{ parameters|join(', ') }}):
    # Imports
    {{ imports|join('\n')|indent(4) }}
    # Body
    {{ body|indent(4) }}
"""
)


def make_input(params: Iterable) -> list[str]:
    return list(map(lambda x: f"{x}: Input[Dataset]", params))


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

    code = TUPLE_FUNCTION_TEMPLATE.render(
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


def create_return_statement(outputs: list[str]) -> tuple[str, str]:

    type_statement = ", ".join(map(lambda x: f"{x}=dsl.Dataset", outputs))

    output_asignment = ", ".join(map(lambda x: f"{x}={x}", outputs))

    output_statement = (
        f'outputs = NamedTuple("outputs", {type_statement})\n'
        f"return outputs({output_asignment})"
    )
    output_type_hint = f'NamedTuple("outputs", {type_statement})'

    return output_statement, output_type_hint


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

    params = list(map(lambda x: f"{x}_dataset", inputs))

    imports = []

    if inputs:
        imports.append(create_import_statement(gs_read_auto))

    imports.append(create_import_statement(function))

    loading_lines = [
        f"{inpt} = gs_read_auto({param}.path)" for param, inpt in zip(params, inputs)
    ]

    loading_statement = "\n".join(loading_lines)

    function_call = create_function_call(
        function_name=function.__name__, args=inputs, kwargs=kwargs, outputs=outputs
    )

    output_statement, output_type_hint = create_return_statement(outputs=outputs)

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

    component_code = create_function_code(
        component_name,
        make_input(params),
        body,
        imports,
        output_type_hint,
        decorator="@component(base_image=MLFLOW_IMAGE)",
    )

    return component_code


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

code += create_function_from_step(**step)

with open("generated_functions.py", "w") as f:
    f.write(code)
