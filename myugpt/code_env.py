import sys
import traceback
from typing import Any, Dict

from myugpt.schema import ProgramInputs, ProgramOutputs, PythonCode, Validation


def run_code(code: str, inputs: Dict[str, Any]) -> Any:
    """
    Run the provided Python code with the given inputs and return the output.
    """
    try:
        # Create a dictionary to serve as the local namespace for the
        # exec function
        local_namespace = inputs.copy()
        exec(code, {}, local_namespace)
        return local_namespace.get("output")
    except Exception:
        print("An error occurred while executing the code:", file=sys.stderr)
        traceback.print_exc()


def validate_code(
    code: PythonCode,
    inputs: ProgramInputs,
    expected_outputs: ProgramOutputs,
) -> Validation:
    """Run the code with the inputs and compare the outputs"""
    # Convert inputs.data from list to dictionary
    inputs_dict = {f"input{i}": inp for i, inp in enumerate(inputs.data)}

    # Run the code
    actual_outputs = run_code(code.data, inputs_dict)

    # Compare the actual outputs with the expected outputs
    if actual_outputs == expected_outputs.data:
        return Validation(outputs=expected_outputs)
    else:
        print("The actual outputs do not match the expected outputs.")
        return Validation(outputs=ProgramOutputs(data=actual_outputs))


if __name__ == "__main__":
    pass
