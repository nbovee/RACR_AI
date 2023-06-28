import os
import sys
import numpy as np
from PIL import Image
import torch
import torch.nn as nn
from torchvision import transforms, models
import time
import pandas as pd
from test_data import test_data_loader as data_loader
import atexit
from collections import OrderedDict 

# from torchvision.models.utils import load_state_dict_from_url
# from typing import Type, Any, Callable, Union, List, Optional, cast

class WrappedModel(nn.Module):
    """Wraps a pretrained model with the features necesarry to perform edge computing tests. Uses pytorch
    hooks to perform benchmarkings, grab intermediate layers, and slice the Sequential to provide input to intermediate layers or exit early. """
    def __init__(self, *args):
        super().__init__(*args)
        self.device = "cpu"
        self.mode = "eval"
        self.dataset = "imagenet"
        self.num_output_layers = None # not needed?
        self.base_input_size = (224, 224)
        self.preprocess = transforms.Compose(
    [
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ]
)
        self.start_layer_index = 0 # inference will be started at this layer.
        self.ignore_layer_index = np.inf # will not perform inference at this layer or above
        self.selected_out = OrderedDict()
        self.pretrained = models.alexnet(pretrained=True)
        self.fhooks = []

        for i,l in enumerate(list(self.pretrained._modules.keys())):
            # if i in self.output_layers:
            print(f"Detected layer layer {i}: {l}")
            self.fhooks.append(getattr(self.pretrained,l).register_forward_hook(self.forward_posthook(l)))
        if self.mode == "eval":
            self.pretrained.eval()
        if self.device == "cuda":
            if torch.cuda.is_available():
                print("Loading Model to CUDA.")
            else:
                print("Loading Model to CPU. CUDA not available.")
                self.device = "cpu"
        self.pretrained.to(self.device)
        self.warmup()
        

    def forward_posthook(self,layer_name):
        """Posthook a layer for output capture and benchmarking."""
        def hook(module, input, output):
            # self.selected_out[layer_name] = output
            pass
        return hook
    
    def forward_prehook(self,layer_name):
        """Prehook a layer for benchmarking."""
        def pre_hook(module, input):
            # return torch.zeros(input[0].shape,dtype=torch.float,device='cuda:0',requires_grad = True)
            return input
        return pre_hook

    def forward(self, x):
        """Wraps the pretrained forward pass to utilize our slicing."""
        net_for_pass = self.pretrained
        # the below could incur some overhead and will need tests
        # net_for_pass = nn.Sequential(*net[start:end])
        with torch.no_grad():
            out = net_for_pass(x)
        return out, self.selected_out

    def parse_input(self, input):
        """Checks if the input is appropriate at the given stage of the network. Does not yet check Tensor shapes for intermediate layers."""
        if isinstance(input, Image.Image):
            if input.size != self.image_size:
                input = input.resize(self.image_size)
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

    def warmup(self, iterations=50):
        if self.device != "cuda":
            print("Warmup not required.")
        else:
            print("Starting warmup.")
            imarray = np.random.rand(*self.image_size, 3) * 255
            with torch.no_grad():
                for i in range(iterations):
                    warmup_image = Image.fromarray(imarray.astype("uint8")).convert("RGB")
                    _ = self.predict(warmup_image)
            print("Warmup complete.")

    def prune_layers():
        """NYE: Trim network layers to attempt usage on low compute power devices, such as early Raspberry Pi models."""
        pass

    def safeClose(self):
        df = pd.DataFrame(data=self.baseline_dict)
        df.to_csv("./test_results/test_results-desktop_cuda.csv")
        torch.cuda.empty_cache()

class Model:
    def __init__(
        self,
        imgnet_classes_fp=os.path.join(
            os.path.realpath(sys.path[0]), "imagenet_classes.txt"
        ),
    ) -> None:
        with open(str(imgnet_classes_fp), "r") as f:
            self.categories = [s.strip() for s in f.readlines()]
        print("Imagenet categories loaded.")
        self.warmup()

if __name__ == "__main__":
    # running as main will test baselines on the running platform
    m = Model(mode="cuda")
    atexit.register(m.safeClose)
    m.baseline_dict = {}
    test_data = data_loader()
    i = 0
    for [data, filename] in test_data.image_list:
        t1 = time.time()
        prediction = m.predict(data)
        print(i)
        i += 1
        m.baseline_dict[filename] = {
            "source": "desktop_cuda",
            "prediction": prediction,
            "inference_time": time.time() - t1,
        }
