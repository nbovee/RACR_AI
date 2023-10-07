import threading


class MasterDict:

    def __init__(self):
        self.lock = threading.Lock()
        self._dict = {}

    def set(self, key: str, value: dict):
        with self.lock:
            if key in self._dict:
                if value.get('layer_information'):
                    self._dict[key]['layer_information'].update(value)
                else:
                    raise ValueError(
                        f"Cannot integrate inference_dict without 'layer_information' field"
                    )
                return
            self._dict[key] = value

    def get(self, key: str):
        with self.lock:
            return self._dict.get(key)

    def __getitem__(self, key: str):
        return self.get(key)

    def __setitem__(self, key: str, newvalue: dict):
        return self.set(key, newvalue)
