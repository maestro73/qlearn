from os import walk
import json
import numpy as np


class BitmexDataReader:

    _file_location = 'bitmex/data/'

    def __init__(self, memory_size=0):
        self.memory_size = memory_size
        self.data = self._data

    @property
    def _data(self):
        data = np.array([])
        for (dirpath, dirnames, filenames) in walk(self._file_location):
            for fn in filenames:
                do = True
                if self.memory_size > 0:
                    if data.size >= self.memory_size:
                        do = False
                        break
                if do is True:
                    with open(f'{self._file_location}/{fn}') as json_file:
                        j_data = np.array(json.loads(json_file.read()))
                        data = np.concatenate([data, j_data])

        if self.memory_size > 0:
            data = data[:self.memory_size]

        print(f'>>> Price memory size: {data.size}')

        return data

    def paginate(self, skip, to):
        return self.data[skip:to]
