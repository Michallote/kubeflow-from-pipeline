from hello import hello_project
from kubeflow_from_pipeline.kfp_generator_declarative import (
    GLOBAL_IMPORTS,
    create_function_from_step,
)


def main():
    step = {
        "name": "raw",
        "function": hello_project,
        "inputs": ["a", "b"],
        "outputs": ["catalog", "segmentation", "tyrany"],
        "kwargs": {"sql_file": "data.sql", "n_iqr": 1.5},
    }

    code = GLOBAL_IMPORTS
    # code += create_function_code(func_name, params, body, imports, output_type_hint)
    # code += create_function_code(
    #     func_name, make_input(params), body, imports, decorator=decorator
    # )

    code += create_function_from_step(
        **step, decorator="@component(base_image=MLFLOW_IMAGE)"
    )

    with open("generated_functions.py", "w") as f:
        f.write(code)


if __name__ == "__main__":
    main()
