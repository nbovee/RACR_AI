class Dataset:
    def has_next(self):
        raise NotImplementedError

    def next(self):
        raise NotImplementedError
