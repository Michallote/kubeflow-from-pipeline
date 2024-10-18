
from typing import Any, NamedTuple
import kfp.dsl as dsl
from collections import namedtuple
from kfp.dsl import Input, Output, Dataset, component
from consts import MLFLOW_IMAGE

@component(base_image=MLFLOW_IMAGE)
def hello_project_raw_catalog_segmentation_tyrany(a_dataset: Input[Dataset], b_dataset: Input[Dataset], catalog_dataset: Output[Dataset], segmentation_dataset: Output[Dataset], tyrany_dataset: Output[Dataset]) -> None:
    # Imports
    from kubeflow_from_pipeline.databases import gs_read_auto
    from kubeflow_from_pipeline.databases import gs_store_auto
    from hello import hello_project
    # Body
    # Inputs Loading
    a = gs_read_auto(a_dataset.path)
    b = gs_read_auto(b_dataset.path)
    # Function Call
    catalog, segmentation, tyrany = hello_project(a, b, sql_file='data.sql', n_iqr=1.5)
    # Outputs Packing
    gs_store_auto(catalog, catalog_dataset.path)
    gs_store_auto(segmentation, segmentation_dataset.path)
    gs_store_auto(tyrany, tyrany_dataset.path)