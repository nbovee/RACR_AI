"""
This file can define multiple custom DataRetriever classes that will be available for future test
cases.
"""


from participant_lib.data_retriever import BaseDataRetriever


class ImagenetFromObserverDL(BaseDataRetriever):
    """
    The BaseDataRetriever class takes care of a lot, but it cannot be used as-is because its 
    class attributes are empty, so it doesn't know where to find its dataset. There are 2 
    ways to tweak the behavior of a custom DataRetriever: overriding the class attributes or
    the methods. NOTE: the init method can be overridden, but it cannot accept any arguments.

    Here is a basic example of how you'd configure a DataRetriever that gets imagenet images
    from the ObserverService:
    """
    # How to get the dataset
    REMOTE_DATASOURCE_ROLE: str = "OBSERVER"
    DATASET_MODULE_NAME: str = "image_datasets"
    DATASET_INSTANCE_FROM_MODULE: str = "imagenet1000_rgb"

    # Other "settings"
    MAX_ITERATIONS: None = None


class MiniImagenetFromObserverDL(ImagenetFromObserverDL):
    """
    This class is almost identical to the one above it; the only difference is that it stops
    after 10 images.
    """
    MAX_ITERATIONS: int = 10
