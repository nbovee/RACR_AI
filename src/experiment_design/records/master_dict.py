import threading


class MasterDict:

    def __init__(self):
        self.lock = threading.RLock()
        self.inner_dict = {}

    def set(self, key: str, value: dict):
        with self.lock:
            if key in self.inner_dict:
                if value.get('layer_information'):
                    self.inner_dict[key]['layer_information'].update(value['layer_information'])
                else:
                    raise ValueError(
                        f"Cannot integrate inference_dict without 'layer_information' field"
                    )
                return
            self.inner_dict[key] = value

    def get(self, key: str):
        with self.lock:
            return self.inner_dict.get(key)

    def update(self, new_info: dict):
        with self.lock:
            for inference_id, layer_data in new_info.items():
                self.set(inference_id, layer_data)

    def __getitem__(self, key: str):
        return self.get(key)

    def __setitem__(self, key: str, newvalue: dict):
        return self.set(key, newvalue)

