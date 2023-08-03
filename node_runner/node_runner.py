from .partitioner.partitioner import Partitioner




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


class Runner:

    def __init__(self, *args, **kwargs):
        super().__init__(*args)
        self.timer = time.perf_counter_ns
        self.info_dict = {}
        self.timing_dict = {}
        self.buffer_dict = {}
        self.device = kwargs.get('mode',"cpu")
        self.number_inferences = 0
        self.mode = "eval"
        self.dataset = "imagenet"
        self.hook_depth = 0
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
        # self.pretrained = models.alexnet(pretrained=True)
        self.pretrained = models.resnet18()
        self.splittable_layer_count = 0
        self.selected_out = OrderedDict() # could be useful for skips
        self.f_hooks = []
        self.f_pre_hooks = []
        # Cant simply profile from a hook due to the possibility of skip connections
        # Similarly, we dont use the global register for hooks because we need more information for our profiling
        self.walk_modules(self.pretrained.children(), 0)
        self.start_layer_index = 0 # inference will never be started below this layer, watch if pruned.
        self.ignore_layer_index = self.splittable_layer_count # will not perform inference at this layer or above, watch if pruned.
        
        if self.mode == "eval":
            self.pretrained.eval()
        if self.device == "cuda":
            if torch.cuda.is_available():
                print("Loading Model to CUDA.")
            else:
                print("Loading Model to CPU. CUDA not available.")
                self.device = "cpu"
        self.pretrained.to(self.device)
        self.warmup(iterations=2)
        self.info_dict['setup'] = {
            'device' : self.device,
            'mode' : self.mode,
            'network' : str(self.pretrained).split('(')[0],
            'dataset' : self.dataset,
            'hook_depth' : self.hook_depth,
            'splittable_layer_count' : self.splittable_layer_count
        }

    
    def enforce_bounds(self, start, end):
        start = self.start_layer_index if start < self.start_layer_index else start
        end = self.ignore_layer_index if end > self.ignore_layer_index else end
        if start >= end:
            raise Exception("Start and End indexes overlap.")
        return start, end

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

    def warmup(self, iterations=50, force = False):
        if self.device != "cuda" and force is not False:
            print("Warmup not required.")
        else:
            print("Starting warmup.")
            with torch.no_grad():
                for i in range(iterations):
                    _ = self(torch.randn(1,3, *self.base_input_size))
            print("Warmup complete.")

    def prune_layers(newlow, newhigh):
        """NYE: Trim network layers. inputs specify the lower and upper layers to REMAIN. Used to attempt usage on low compute power devices, such as early Raspberry Pi models."""
        pass

    def safeClose(self):
        df = pd.DataFrame(data=self.timing_dict)
        df2 = pd.DataFrame(data=self.info_dict)
        df = df.transpose()
        name = f"./test_results/test_results-desktop_cuda{time.time()}"
        df2.to_json(name + ".json")
        df.to_csv(name + ".csv", index=False)
        torch.cuda.empty_cache()

if __name__ == "__main__":
    # running as main will test baselines on the running platform
    m = WrappedModel(mode="cuda")
    atexit.register(m.safeClose)
    test_data = data_loader()
    i = 0
    for [data, filename] in test_data.image_list:
        t1 = time.time()
        prediction = m.predict(data)
        print(i)
        i += 1
        m.timing_dict[filename] = {
            "source": "desktop_cuda",
            "prediction": prediction,
            "inference_time": time.time() - t1,
        }
