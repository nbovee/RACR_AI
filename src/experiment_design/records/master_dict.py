import pickle
import threading
import pandas as pd
from rpyc.utils.classic import obtain


class MasterDict:

    def __init__(self):
        self.lock = threading.RLock()
        self.inner_dict = {}

    def set(self, key: str, value: dict):
        with self.lock:
            if key in self.inner_dict:
                if value.get('layer_information'):
                    layer_info = value["layer_information"]
                    layer_info = {k: v for k, v in layer_info.items() if v["inference_time"] is not None}
                    self.inner_dict[key]['layer_information'].update(layer_info)
                else:
                    raise ValueError(
                        f"Cannot integrate inference_dict without 'layer_information' field"
                    )
                return
            self.inner_dict[key] = value

    def get(self, key: str):
        with self.lock:
            return self.inner_dict.get(key)

    def update(self, new_info: dict, by_value=True):
        if by_value:
            new_info = obtain(new_info)
        with self.lock:
            for inference_id, layer_data in new_info.items():
                self.set(inference_id, layer_data)

    def get_total_inference_time(self, inference_id: str) -> int:
        inf_data = self.inner_dict[inference_id]
        layer_times = [
            layer["inference_time"]
            for layer in inf_data["layer_information"].values()
            if layer["inference_time"]
        ]
        return int(sum(layer_times))

    def get_split_layer(self, inference_id: str):
        inf_data = self.inner_dict[inference_id]
        layer_ids = sorted(list(inf_data["layer_information"].keys()))
        start_node = inf_data["layer_information"][0]["completed_by_node"]
        for layer_id in layer_ids:
            if inf_data["layer_information"][layer_id]["completed_by_node"] != start_node:
                return layer_id

    def to_dataframe(self) -> pd.DataFrame:
        flattened_data = []
        layer_attrs = []

        for superfields in self.inner_dict.values():
            inf_id = superfields["inference_id"]
            split_layer = self.get_split_layer(inf_id)
            total_time_ns = self.get_total_inference_time(inf_id)
            for subdict in superfields["layer_information"].values():
                layer_id = subdict.pop("layer_id")
                if not layer_attrs:
                    layer_attrs = [key for key in subdict.keys()]
                row = (inf_id, layer_id, split_layer, total_time_ns, *[subdict[k] for k in layer_attrs])
                flattened_data.append(row)

        flattened_data.sort(key=lambda tup: (tup[0], tup[1]))

        columns = ["inference_id", "layer_id", "split_layer", "total_inference_time_ns", *layer_attrs]
        df = pd.DataFrame(flattened_data, columns=columns)

        return df

    def to_pickle(self):
        return pickle.dumps(self.inner_dict)

    def __getitem__(self, key: str):
        return self.get(key)

    def __setitem__(self, key: str, newvalue: dict):
        return self.set(key, newvalue)


if __name__ == "__main__":
    print("Running test.")
    from src.app_api.utils import get_repo_root

    test_dict = get_repo_root() / "MyData" / "TestResults" / "alexnetsplit__2023-10-25T220600.pkl"
    assert test_dict.exists()
    print("test dict exists.")

    with open(test_dict, "rb") as file:
        test_dict = pickle.load(file)

    sample = test_dict[list(test_dict.keys())[7]]
    print(f"Unpickled test dict. Sample value: {sample}")

    master_dict = MasterDict()
    master_dict.inner_dict = test_dict

    test_df = master_dict.to_dataframe()
    print(test_df.head(45))

