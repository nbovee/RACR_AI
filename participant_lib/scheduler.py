

from participant_lib.dataloader import BaseDataLoader
from participant_lib.model_hooked import WrappedModel


class BaseScheduler:
    """
    The scheduler has a more general role than the name suggests. A better name for this component
    for the future might be "runner" or something similar. The scheduler takes a DataLoader and a 
    WrappedModel as parameters to __init__ so it can essentially guide the participating node 
    through its required sequence of tasks.
    """
    
    dataloader: BaseDataLoader
    model: WrappedModel

    def __init__(self, dataloader: BaseDataLoader, model: WrappedModel):
        self.dataloader = dataloader
        self.model = model

    def 
