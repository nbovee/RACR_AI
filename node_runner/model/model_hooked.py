import os
import sys
import numpy as np
from PIL import Image
import torch
import torch.nn as nn
from torchvision import transforms, models
import time
import pandas as pd
# from test_data import test_data_loader as data_loader
import atexit
from collections import OrderedDict

class HookExitException(Exception):
    """Exception to early exit from inference in naive running."""
    pass

class WrappedModel(nn.Module):
    """Wraps a pretrained model with the features necesarry to perform edge computing tests. Uses pytorch
    hooks to perform benchmarkings, grab intermediate layers, and slice the Sequential to provide input to intermediate layers or exit early. """

    def __init__(self, *args, **kwargs):
        print(*args)
        super().__init__(*args)
        self.timer = time.perf_counter_ns
        self.timing_dict = {}
        self.buffer_dict = {}
        self.device = kwargs.get('mode',"cpu")
        self.number_inferences = 0
        self.mode = "eval"
        self.hook_depth = 1
        self.base_input_size = (224, 224)

        self.pretrained = kwargs.pop('pretrained', models.alexnet(pretrained=True))
        self.splittable_layer_count = 0
        self.selected_out = OrderedDict() # could be useful for skips
        self.f_hooks = []
        self.f_pre_hooks = []
        # Cant simply profile from a hook due to the possibility of skip connections
        # Similarly, we dont use the global register for hooks because we need more information for our profiling
        self.walk_modules(self.pretrained.children(), 0)
        # values the hooks watch
        self.current_module_stop_index = None
        self.current_module_index = None
        self.start_layer_index = 0 # inference will never be started below this layer, watch if pruned.
        self.max_ignore_layer_index = self.splittable_layer_count # will not perform inference at this layer or above, watch if pruned.
        
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
        
    def walk_modules(self, module_generator, depth):
        '''Recursively walks and marks Modules for hooks in a DFS. Most NN have an intended or intuitive depth to split at, but it is not obvious to the naive program.'''
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
            print(f"val. {self.current_module_index}")
        return pre_hook
        
    def forward_posthook(self, layer_index, layer_name, input_shape, **kwargs):
        """Posthook a layer for output capture and benchmarking."""
        def hook(module, args, output):
            self.buffer_dict[layer_index] += self.timer()            
            print(f"stop at layer: {self.current_module_stop_index -1}")
            print(f"L{layer_index}-{layer_name} returned.")
            self.current_module_index += 1
            # if self.current_module_index <= 
            # if current layer is stop index -1 , raise WrappedModel.HookExitException(output)
        return hook
    
    def enforce_bounds(self, start, end):
        start = self.start_layer_index if start < self.start_layer_index else start
        end = self.max_ignore_layer_index if end > self.max_ignore_layer_index else end
        if start >= end:
            raise Exception("Start and End indexes overlap.")
        return start, end

    def forward(self, x, start = 0, end = np.inf):
        """Wraps the pretrained forward pass to utilize our slicing."""
        start, end = self.enforce_bounds(start, end)
        self.current_module_stop_index = end
        self.current_module_index = 0
        
        # timing values generated by hooks go to an buffer dict, which is cleard after each complete pass 
        try:
            if self.mode != "train":
                with torch.no_grad():
                    out = self.pretrained(x)
            else:
                out = self.pretrained(x)
        except HookExitException as e:
            print("Exit early from hook.")
            out = e.result

        self.timing_dict[self.number_inferences] = self.buffer_dict
        self.buffer_dict = {}
        self.number_inferences += 1
        self.current_module_stop_index = None
        self.current_module_index = None
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
        raise NotImplementedError

    def safeClose(self):
        torch.cuda.empty_cache()

if __name__ == "__main__":
    # running as main will test baselines on the running platform
    m = WrappedModel(mode="cuda")
    atexit.register(m.safeClose)
    m
    # test_data = data_loader()
    # i = 0
    # for [data, filename] in test_data.image_list:
    #     t1 = time.time()
    #     prediction = m.predict(data)
    #     print(i)
    #     i += 1
    #     m.timing_dict[filename] = {
    #         "source": "desktop_cuda",
    #         "prediction": prediction,
    #         "inference_time": time.time() - t1,
    #     }
