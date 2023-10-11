"""
The two executors defined here implement basic split inference behavior between a single "client"
node and a single "edge" node. This behavior matches that of one of the first test cases we used
to validate the test system: a client node (perhaps an autonomous vehicle) initiates an inference,
then passes the intermediary data off to a nearby edge server for completion.

We also define a subclass of BaseDelegator to run the observer node.
"""

import uuid
from experiment_design.partitioners.partitioner import Partitioner
from experiment_design.rpc_services.participant_service import ParticipantService
import experiment_design.tasks.tasks as tasks
from runner import BaseDelegator, BaseExecutor


class ClientExecutor(BaseExecutor):
    """
    To define the way our client behaves, there are only three parts of the BaseExecutor class we
    have to overwrite:
        1.) The `partners` class attribute should list the names of the nodes we want to handshake
            with before the executor starts
        2.) The inference_sequence_per_input method tells the node to try a split at each possible
            layer for each input image it receives using the "cycle" type partitioner class,
            sending a SimpleInferenceTask to the "EDGE" node each time
        3.) The `on_finish` method should send a `FinishSignalTask` instance to the edge node's 
            inbox so it knows it's done after it has finished all the inference tasks we sent it

    We also add a DOWNSTREAM_PARTNER class attribute for readability/adaptability.

    When the experiment runs, this executor will actually respond to an instance of 
    `InferOverDatasetTask`, but because the base class's corresponding `infer_dataset` method just
    calls `inference_sequence_per_input` repeatedly, we don't have to change it directly.
    """

    DOWNSTREAM_PARTNER = "EDGE"

    partners: list[str] = ["OBSERVER", "EDGE"]

    def inference_sequence_per_input(self, task: tasks.SingleInputInferenceTask):
        assert self.model is not None
        input = task.input
        inference_id = task.inference_id if task.inference_id is not None else str(uuid.uuid4())
        splittable_layer_count = self.model.splittable_layer_count
        partitioner = Partitioner.create("cycle", splittable_layer_count)

        current_split_layer = 0
        while current_split_layer < splittable_layer_count:
            current_split_layer = partitioner()
            start, end = 0, current_split_layer
            out = self.model.forward(
                input, inference_id, start=start, end=end, by_node=self.node.node_name
            )
            downstream_task = tasks.SimpleInferenceTask(
                self.node.node_name, out, inference_id, start_layer=end+1
            )
            downstream_partner = self.node.get_connection(self.DOWNSTREAM_PARTNER)
            assert isinstance(downstream_partner, ParticipantService)
            downstream_partner.give_task(downstream_task)

    def on_finish(self):
        downstream_finish_signal = tasks.FinishSignalTask(self.node.node_name)
        downstream_partner = self.node.get_connection(self.DOWNSTREAM_PARTNER)
        assert isinstance(downstream_partner, ParticipantService)
        downstream_partner.give_task(downstream_finish_signal)


class EdgeExecutor(BaseExecutor):
    """
    Defining our edge node's behavior is much simpler because it only has to respond to instances
    of the `SimpleInferenceTask` class, which is already defined in the BaseExecutor class. The 
    edge node has no "decisions" to make; the client gives it all the parameters necessary to 
    perform its inference. All we need to do is overwrite the list of partners it will handshake
    with during setup.
    """
    partners: list[str] = ["OBSERVER", "CLIENT"]


class ClientEdgeDelegator(BaseDelegator):
    """
    For the observer node, we just need to overwrite two things:
        1.) The `partners` class attribute tells the observer which participant nodes to wait for 
        2.) The `playbook` class attribute tells it which tasks should be assigned to each
            participant before starting the experiment
    """
    partners: list[str] = ["CLIENT", "EDGE"]
    playbook: dict[str, list[tasks.Task]] = {
        # The playbook has no entry for EDGE since it gets its tasks from CLIENT in this test case
        "CLIENT": [
            # First, CLIENT runs its inference_sequence_per_input method on each dataset instance
            tasks.InferOverDatasetTask("OBSERVER", "imagenet", "imagenet10_rgb"),
            # Then it receives the finish signal, which prompts it to send a finish signal to 
            # EDGE as well, before stopping itself
            tasks.FinishSignalTask("OBSERVER")
        ]
    }
