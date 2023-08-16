from partitioner.linreg_partitioner import RegressionPartitioner
from model.model_hooked import WrappedModel

import torch
# this module reads configs and sets up the modules and local rpc




def parse_input(self, input):
    """Checks if the input is appropriate at the given stage of the network. Does not yet check Tensor shapes for intermediate layers."""
    if isinstance(input, Image.Image):
        if input.size != self.base_input_size:
            input = input.resize(self.base_input_size)
        input_tensor = self.preprocess(input)
        input_tensor = input_tensor.unsqueeze(0)
    elif isinstance(input, torch.Tensor):
        input_tensor = input
    if (
        torch.cuda.is_available()
        and self.mode == "cuda"
        and input_tensor.device != self.mode
    ):
        input_tensor = input_tensor.to(self.mode)
    return input_tensor

def parse_output(self, predictions):
    """Take the final output tensor of the wrapped model and map it to its appropriate human readable results."""
    probabilities = torch.nn.functional.softmax(predictions[0], dim=0)
    # Show top categories per image
    top1_prob, top1_catid = torch.topk(probabilities, 1)
    prediction = self.categories[top1_catid]
    return prediction

def safeClose(self):
    pass

if __name__ == "__main__":
    global master_dictionary
    master_dictionary = dict()
    m = WrappedModel(dict = master_dictionary, depth = 2)
    linreg = RegressionPartitioner(m.splittable_layer_count)
    linreg.create_data(m)
    linreg.update_regression()
    m((torch.randn(1, *m.base_input_size)))
