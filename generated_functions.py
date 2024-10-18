
from typing import Any, NamedTuple
import kfp.dsl as dsl
from collections import namedtuple
from kfp.dsl import Input, Output, Dataset, component
from consts import MLFLOW_IMAGE


def multiply_numbers(a, b) -> NamedTuple("outputs", catalog_full=dsl.Dataset):
    # Imports
    from src.databases.load_gs_data import gs_read_df
    from src.databases.data_collection import combine_catalog_sources
    # Body
    

    catalog_full = combine_catalog_sources(a, b)

    outputs = namedtuple("outputs", catalog_full=dsl.Dataset)
    return outputs(catalog_full=catalog_full)

@component(base_image=MLFLOW_IMAGE)
def multiply_numbers(a: Input[Dataset], b: Input[Dataset]) -> Any:
    # Imports
    from src.databases.load_gs_data import gs_read_df
    from src.databases.data_collection import combine_catalog_sources
    # Body
    

    catalog_full = combine_catalog_sources(a, b)

    outputs = namedtuple("outputs", catalog_full=dsl.Dataset)
    return outputs(catalog_full=catalog_full)

@component(base_image=MLFLOW_IMAGE)
def hello_project_raw_catalog(a_dataset: Input[Dataset], b_dataset: Input[Dataset]) -> NamedTuple("outputs", catalog=dsl.Dataset):
    # Imports
    from kubeflow_from_pipeline.databases import gs_read_auto
    from hello import hello_project
    # Body
    # Inputs Loading
    a = gs_read_auto(a_dataset.path)
    b = gs_read_auto(b_dataset.path)
    # Function Call
    catalog = hello_project(a, b, sql_file='data.sql', n_iqr=1.5)
    # Outputs Packing
    outputs = NamedTuple("outputs", catalog=dsl.Dataset)
    return outputs(catalog=catalog)
