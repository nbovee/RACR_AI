import importlib

class ParticipantNode:
    """
    Takes the user-defined Model, Runner, and DataLoader classes as parameters and bundles them
    into a composition class to decouple their functionality from the RPC calls.
    """


module = importlib.import_module("user_module")

Model, Runner, DataLoader = tuple(
    [getattr(module, class_name) for class_name in ("Model", "Runner", "DataLoader")]
)


