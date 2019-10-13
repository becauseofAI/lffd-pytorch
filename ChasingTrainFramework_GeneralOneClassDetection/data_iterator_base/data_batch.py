# coding: utf-8


class DataBatch:
    def __init__(self, torch_module):
        self._data = []
        self._label = []
        self.torch_module = torch_module

    def append_data(self, new_data):
        self._data.append(self.__as_tensor(new_data))

    def append_label(self, new_label):
        self._label.append(self.__as_tensor(new_label))

    def __as_tensor(self, in_data):
        return self.torch_module.from_numpy(in_data)

    @property
    def data(self):
        return self._data

    @property
    def label(self):
        return self._label
