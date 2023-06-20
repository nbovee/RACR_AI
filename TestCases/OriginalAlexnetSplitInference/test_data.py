from data_wrapper import data_wrapper
from PIL import Image
import glob
import pathlib
import sys


test_img_dir = pathlib.Path(sys.path[0]) / "test"


class test_data_loader(data_wrapper):
    def __init__(self, img_directory=test_img_dir):
        self.image_list = []
        self.load_data(img_directory)

    def has_next(self):
        return bool(len(self.image_list))

    def next(self):
        if self.has_next() is None:
            return None
        else:
            [val, filename] = self.image_list.pop()
            print(
                f'Testing on image "{pathlib.Path(filename).name}"'
                + f"\t\t({len(self.image_list)} remaining)"
            )
            for i in range(1, 21):
                # hardcoded split layers for AlexNet - no full processing yet.
                # Layer 0 would be full server, 21 would be full client
                yield [val, i, filename]

    def load_data(self, path, fn_glob_pattern="*.JPEG"):
        max_images = 1000
        self.image_list.clear()
        for image in glob.iglob(f"{str(path)}/{fn_glob_pattern}"):
            self.image_list.append([Image.open(image).convert("RGB"), image])
            if len(self.image_list) >= max_images:
                break
