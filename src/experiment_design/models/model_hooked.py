"""model_hooked module"""

import atexit
import copy
import logging
import time
from collections import OrderedDict

import numpy as np
import torch
from PIL import Image
from torchinfo import summary

from .model_config import read_model_config
from .model_selector import model_selector

atexit.register(torch.cuda.empty_cache)
logger = logging.getLogger("tracr_logger")


class HookExitException(Exception):
    """Exception to early exit from inference in naive running."""

    def __init__(self, out, *args: object) -> None:
        super().__init__(*args)
        self.result = out


class WrappedModel(torch.nn.Module):
    """Wraps a pretrained model with the features necesarry to perform edge computing tests.
    Uses pytorch hooks to perform benchmarkings, grab intermediate layers, and slice the
    Sequential to provide input to intermediate layers or exit early.
    """

    layer_template_dict = {
        "layer_id": None,  # this one may prove unneeded
        "class": None,
        "inference_time": None,
        "parameters": None,
        "parameter_bytes": None,
        # "precision": None, precision is not technically per layer, disabled for now
        "cpu_cycles_used": None,
        "watts_used": None,
    }

    def __init__(self, *args, config_path=None, master_dict=None, **kwargs):
        logger.debug(f"{args=}")
        super().__init__(*args)
        self.timer = time.perf_counter_ns
        self.master_dict = master_dict  # this should be the externally accessible dict
        self.inference_dict = (
            {}
        )  # collation dict for the current partition of a given inference
        self.forward_dict = {}  # dict for the results from the current forward pass
        # assigns config vars to the wrapper
        self.__dict__.update(read_model_config(config_path))
        self.model = model_selector(self.model_name)
        self.splittable_layer_count = 0
        self.selected_out = OrderedDict()  # could be useful for skips
        self.f_hooks = []
        self.f_pre_hooks = []
        # run torchinfo here to get parameters/flops/mac for entry into dict
        self.torchinfo = summary(self.model, (1, *self.input_size), verbose=0)
        self.walk_modules(
            self.model.children(), 1
        )  # depth starts at 1 to match torchinfo depths
        del self.torchinfo
        self.forward_dict_empty = copy.deepcopy(self.forward_dict)
        # ---- class scope values that the hooks and forward pass use ----
        self.current_module_start_index = None
        self.current_module_stop_index = None
        self.current_module_index = None
        self.banked_input = None

        # layers before this index are not added to tracking dictionary,
        # as they are not based upon the given input tensor

        # will not perform inference at this layer or above, watch if pruned.
        self.max_ignore_layer_index = self.splittable_layer_count - 1

        if self.mode == "eval":
            self.model.eval()
        if self.device == "cuda":
            if torch.cuda.is_available():
                logger.info("Loading Model to CUDA.")
            else:
                logger.info("Loading Model to CPU. CUDA not available.")
                self.device = "cpu"
        self.model.to(self.device)
        self.warmup(iterations=2)

    def walk_modules(self, module_generator, depth):
        """Recursively walks and marks Modules for hooks in a DFS. Most NN have an
        intended or intuitive depth to split at, but it is not obvious to the naive program.
        """
        for child in module_generator:
            childname = str(child).split("(", maxsplit=1)[0]
            if len(list(child.children())) > 0 and depth < self.depth:
                # either has children we want to look at, or is max depth
                logger.debug(
                    f"{'-'*depth}Module {childname} "
                    "with children found, hooking children instead of module."
                )
                self.walk_modules(child.children(), depth + 1)
                logger.debug(f"{'-'*depth}End of Module {childname}'s children.")
            elif isinstance(child, torch.nn.Module):
                # if not iterable/too deep, we have found a layer to hook
                for layer in self.torchinfo.summary_list:
                    if layer.layer_id == id(child):
                        self.forward_dict[self.splittable_layer_count] = copy.deepcopy(
                            WrappedModel.layer_template_dict
                        ).update(
                            {
                                "depth": depth,
                                "layer_id": self.splittable_layer_count,
                                "class": layer.class_name,
                                # "precision": None,
                                "parameters": layer.num_params,
                                "parameter_bytes": layer.param_bytes,
                                "input_size": layer.input_size,
                                "output_size": layer.output_size,
                                "output_bytes": layer.output_bytes,
                            }
                        )

                self.f_hooks.append(
                    child.register_forward_pre_hook(
                        self.forward_prehook(
                            self.splittable_layer_count,
                            childname,
                            (0, 0),
                        ),
                        with_kwargs=False,
                    )
                )
                self.f_pre_hooks.append(
                    child.register_forward_hook(
                        self.forward_posthook(
                            self.splittable_layer_count,
                            childname,
                            (0, 0),
                        ),
                        with_kwargs=False,
                    )
                )
                logger.debug(
                    f"{'-'*depth}Layer {self.splittable_layer_count}: "
                    f"{childname} had hooks applied."
                )
                # back hooks left out for now
                self.splittable_layer_count += 1

    def forward_prehook(self, layer_index, layer_name, input_shape):
        """Prehook a layer for benchmarking."""

        def pre_hook(module, layer_input):  # hook signature format is required
            if self.log and (
                self.current_module_index >= self.current_module_start_index
            ):
                self.forward_dict[layer_index]["inference_time"] = -self.timer()
            # store input until the correct layer arrives
                
            # NTS rewrite banked input to also handle skip connections
            if self.current_module_index == 0 and self.current_module_start_index > 0:
                self.banked_input = copy.deepcopy(layer_input)
                return torch.randn(1, *self.input_size)
            # swap correct input back in now that we are at the right layer
            elif (
                self.banked_input is not None
                and self.current_module_index == self.current_module_start_index
            ):
                layer_input = self.banked_input
                self.banked_input = None
                return layer_input

        return pre_hook

    def forward_posthook(self, layer_index, layer_name, input_shape, **kwargs):
        """Posthook a layer for output capture and benchmarking."""

        def hook(
            module, layer_input, layer_output
        ):  # hook signature format is required
            if (
                self.log
                and self.current_module_index >= self.current_module_start_index
            ):
                self.forward_dict[layer_index]["inference_time"] += self.timer()
            if (
                layer_index >= self.current_module_stop_index - 1
                and layer_index < self.max_ignore_layer_index - 1
            ):
                raise HookExitException(layer_output)
            self.current_module_index += 1
            logger.debug(f"stop at layer: {self.current_module_stop_index -1}")
            logger.debug(f"L{layer_index}-{layer_name} returned.")

        return hook

    def forward(self, x, inference_id=None, start=0, end=np.inf, log=True):
        """Wraps the model forward pass to utilize our slicing."""
        end = self.splittable_layer_count if end == np.inf else end

        # set values for the hooks to see
        self.log = log
        self.current_module_stop_index = end
        self.current_module_index = 0
        self.current_module_start_index = start

        # prepare inference_id for storing results
        _inference_id = "unlogged" if inference_id is None else inference_id
        if len(str(_inference_id).split(".")) > 1:
            suffix = int(str(_inference_id).rsplit(".", maxsplit=1)[-1]) + 1
        else:
            suffix = 0
        _inference_id = str(str(_inference_id).split(".", maxsplit=1)[0]) + f".{suffix}"
        self.inference_dict["inference_id"] = _inference_id
        logger.info(f"{_inference_id} id beginning.")
        # actually run the forward pass
        try:
            if self.mode != "train":
                with torch.no_grad():
                    out = self.model(x)
            else:
                out = self.model(x)
        except HookExitException as e:
            logger.debug("Exit early from hook.")
            out = e.result
            for i in range(self.current_module_stop_index, self.splittable_layer_count):
                del self.forward_dict[i]

        # process and clean dicts before leaving forward
        self.inference_dict["layer_information"] = self.forward_dict
        if log:
            self.master_dict[
                str(_inference_id).split(".", maxsplit=1)[0]
            ] = copy.deepcopy(
                self.inference_dict
            )  # only one deepcopy needed
        self.inference_dict = {}
        self.forward_dict = copy.deepcopy(self.forward_dict_empty)

        # reset hook variables
        self.current_module_stop_index = None
        self.current_module_index = None
        logger.info(f"{_inference_id} end.")
        return out

    def parse_input(self, layer_input):
        """Checks if the input is appropriate at the given stage of the network.
        Does not yet check Tensor shapes for intermediate layers."""
        if isinstance(layer_input, Image.Image):
            if layer_input.size != self.input_size:
                layer_input = layer_input.resize(self.input_size)
            input_tensor = self.preprocess(layer_input)
            input_tensor = input_tensor.unsqueeze(0)
        elif isinstance(layer_input, torch.Tensor):
            input_tensor = layer_input
        if (
            torch.cuda.is_available()
            and self.mode == "cuda"
            and input_tensor.device != self.mode
        ):
            input_tensor = input_tensor.to(self.mode)
        return input_tensor

    def warmup(self, iterations=50, force=False):
        """runs specified passes on the nn to warm up gpu if enabled"""
        if self.device != "cuda" and force is not False:
            logger.info("Warmup not required.")
        else:
            logger.info("Starting warmup.")
            with torch.no_grad():
                for _ in range(iterations):
                    self(torch.randn(1, *self.input_size), log=False)
            logger.info("Warmup complete.")

    def prune_layers(self, newlow, newhigh):
        """NYE: Trim network layers. inputs specify the lower and upper layers to REMAIN.
        Used to attempt usage on low compute power devices, such as early Raspberry Pi models.
        """
        raise NotImplementedError
