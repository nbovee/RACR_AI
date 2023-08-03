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

class WrappedModel(nn.Module):
    """Wraps a pretrained model with the features necesarry to perform edge computing tests. Uses pytorch
    hooks to perform benchmarkings, grab intermediate layers, and slice the Sequential to provide input to intermediate layers or exit early. """
    def __init__(self, *args, **kwargs):
        print(*args)
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
        
    def walk_modules(self, module_generator, depth):
        '''Recursively walks and marks Modules for hooks in a DFS'''
        for child in module_generator:
            if len(list(child.children())) > 0 and depth < self.hook_depth:
                # either has children we want to look at, or is max depth
                print(f"Module {str(child).split('(')[0]} with children found, hooking children instead of module.")
                self.walk_modules(child.children(), depth + 1)
            elif isinstance(child, nn.Module):
            # if not iterable/too deep, we have found a layer to hook
                print(f"Layer {self.splittable_layer_count}: {str(child).split('(')[0]} hooks applied.")
                # there must be a better way to get names but not needed atm
                self.f_hooks.append(child.register_forward_pre_hook(self.forward_prehook(self.splittable_layer_count, str(child).split('(')[0], (0, 0)), with_kwargs=False))
                self.f_pre_hooks.append(child.register_forward_hook(self.forward_posthook(self.splittable_layer_count, str(child).split('(')[0], (0, 0)), with_kwargs=False))
                # back hooks left out for now
                self.splittable_layer_count += 1

    def forward_prehook(self, layer_index, layer_name, input_shape):
        """Prehook a layer for benchmarking."""
        def pre_hook(module, args):
            self.buffer_dict[layer_index] = -self.timer()
            print(f"L{layer_index}-{layer_name} called.")
        return pre_hook
        
    def forward_posthook(self, layer_index, layer_name, input_shape, **kwargs):
        """Posthook a layer for output capture and benchmarking."""
        def hook(module, args, output):
            self.buffer_dict[layer_index] += self.timer()
            print(f"L{layer_index}-{layer_name} returned.")
        return hook
    
    def enforce_bounds(self, start, end):
        start = self.start_layer_index if start < self.start_layer_index else start
        end = self.ignore_layer_index if end > self.ignore_layer_index else end
        if start >= end:
            raise Exception("Start and End indexes overlap.")
        return start, end

    def forward(self, x, start = 0, end = np.inf):
        """Wraps the pretrained forward pass to utilize our slicing."""
        start, end = self.enforce_bounds(start, end)
        net_for_pass = self.pretrained
        # Alexnet can be easily set up into a sliceable structure, but more complex networks may not be possbile without custom forward passes.
        # For benchmarking, it would be possible to await a Tensor from a pre-hook, but that is not practical for actual use.
        # Need discussing with Algo Team.
        # net_for_pass = nn.Sequential(*net_for_pass[start:end])
        
        # timing values generated by hooks go to an buffer dict, which is cleard after each complete pass 
        with torch.no_grad():
            out = net_for_pass(x)
        self.timing_dict[self.number_inferences] = self.buffer_dict
        self.buffer_dict = {}
        self.number_inferences += 1
        return out

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

# class Model:
#     def __init__(
#         self,
#         imgnet_classes_fp=os.path.join(
#             os.path.realpath(sys.path[0]), "imagenet_classes.txt"
#         ),
#     ) -> None:
#         with open(str(imgnet_classes_fp), "r") as f:
#             self.categories = [s.strip() for s in f.readlines()]
#         print("Imagenet categories loaded.")
#         self.warmup()

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
