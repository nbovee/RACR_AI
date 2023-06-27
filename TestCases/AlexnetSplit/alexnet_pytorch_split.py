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


image_size = (224, 224)
model = None
max_layers = 21

preprocess = transforms.Compose(
    [
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ]
)


class SplitAlex(models.AlexNet):
    # almost exactly like pytorch AlexNet, but we cannot split out of a Sequential some
    # ModuleList is used instead
    def __init__(self, num_classes: int = 1000, dropout: float = 0.5) -> None:
        super().__init__()

        self.features = nn.ModuleList(
            [
                nn.Conv2d(3, 64, kernel_size=11, stride=4, padding=2),  # 0
                nn.ReLU(inplace=True),  # 1
                nn.MaxPool2d(kernel_size=3, stride=2),  # 2
                nn.Conv2d(64, 192, kernel_size=5, padding=2),  # 3
                nn.ReLU(inplace=True),  # 4
                nn.MaxPool2d(kernel_size=3, stride=2),  # 5
                nn.Conv2d(192, 384, kernel_size=3, padding=1),  # 6
                nn.ReLU(inplace=True),  # 7
                nn.Conv2d(384, 256, kernel_size=3, padding=1),  # 8
                nn.ReLU(inplace=True),  # 9
                nn.Conv2d(256, 256, kernel_size=3, padding=1),  # 10
                nn.ReLU(inplace=True),  # 11
                nn.MaxPool2d(kernel_size=3, stride=2),  # 12
            ]
        )

        self.avgpool = nn.AdaptiveAvgPool2d((6, 6))  # 13 + flatten
        self.classifier = nn.ModuleList(
            [
                nn.Dropout(p=dropout),  # 14
                nn.Linear(256 * 6 * 6, 4096),  # 15
                nn.ReLU(inplace=True),  # 16
                nn.Dropout(p=dropout),  # 17
                nn.Linear(4096, 4096),  # 18
                nn.ReLU(inplace=True),  # 19
                nn.Linear(4096, num_classes),  # 20
            ]
        )

    def forward(self, x: torch.Tensor, start_layer=0, end_layer=np.inf) -> torch.Tensor:
        # end layer will not be processed
        prints = False
        # if start_layer != 0:
        #     prints = True
        active_layer = start_layer
        while active_layer < min(len(self.features), end_layer):
            if prints:
                print(f"features{active_layer}")
            x = self.features[active_layer].forward(x)
            active_layer += 1
        while active_layer == len(self.features) and active_layer != end_layer:
            if prints:
                print(f"pool{active_layer}")
                # print(x)
            x = self.avgpool(x)
            x = torch.flatten(x, 1)
            active_layer += 1
        while active_layer < min(
            len(self.features) + len(self.classifier) + 1, end_layer
        ):
            if prints:
                print(f"class{active_layer-len(self.features)-1}")
            # fix magic offset later
            x = self.classifier[active_layer - (len(self.features) + 1)].forward(x)
            active_layer += 1
        return x


class Model:
    def __init__(
        self,
        mode="cpu",
        imgnet_classes_fp=os.path.join(
            os.path.realpath(sys.path[0]), "imagenet_classes.txt"
        ),
    ) -> None:
        global model
        self.mode = mode
        model = SplitAlex()
        model.load_state_dict(models.alexnet(pretrained=True).state_dict())
        model.eval()
        self.max_layers = max_layers
        if torch.cuda.is_available() and self.mode == "cuda":
            print("Loading Model to CUDA.")
        else:
            print("Loading Model to CPU.")
        model.to(self.mode)
        with open(str(imgnet_classes_fp), "r") as f:
            self.categories = [s.strip() for s in f.readlines()]
        print("Imagenet categories loaded.")
        self.warmup()

    def predict(self, payload, start_layer=0, end_layer=np.inf):
        if isinstance(payload, Image.Image):
            if payload.size != image_size:
                payload = payload.resize(image_size)
            input_tensor = preprocess(payload)
            input_tensor = input_tensor.unsqueeze(0)
        elif isinstance(payload, torch.Tensor):
            input_tensor = payload
        if (
            torch.cuda.is_available()
            and self.mode == "cuda"
            and input_tensor.device != self.mode
        ):
            input_tensor = input_tensor.to(self.mode)
        with torch.no_grad():
            predictions = model(
                input_tensor, start_layer=start_layer, end_layer=end_layer
            )
        if end_layer < 21:  # fix magic number
            return predictions
        else:
            probabilities = torch.nn.functional.softmax(predictions[0], dim=0)
            # Show top categories per image
            top1_prob, top1_catid = torch.topk(probabilities, 1)
            # print(top1_catid)
            prediction = self.categories[top1_catid]
            return prediction

    def warmup(self, iterations=50):
        if self.mode != "cuda":
            print("Warmup not required.")
        else:
            print("Starting warmup.")
            imarray = np.random.rand(*image_size, 3) * 255
            for i in range(iterations):
                warmup_image = Image.fromarray(imarray.astype("uint8")).convert("RGB")
                _ = self.predict(warmup_image)
            print("Warmup complete.")

    def safeClose(self):
        df = pd.DataFrame(data=self.baseline_dict)
        df.to_csv("./test_results/test_results-desktop_cuda.csv")
        torch.cuda.empty_cache()


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
