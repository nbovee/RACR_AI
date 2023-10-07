from dataclasses import dataclass, field

@dataclass(order=True)
class Task:

    priority: int
    task_type: str = field(compare=False)
    from_node: str = field(compare=False)

