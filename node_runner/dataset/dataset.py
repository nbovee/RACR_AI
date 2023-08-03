class Dataset:
    def has_next(self):
        raise NotImplementedError

    def next(self):
        raise NotImplementedError

    def normalize(self):
        self.preprocess = transforms.Compose(
    [
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ]
)