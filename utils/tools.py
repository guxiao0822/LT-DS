from collections import Counter
import numpy as np

class ForeverDataIterator:
    """A data iterator that will never stop producing data"""
    def __init__(self, data_loader):
        self.data_loader = data_loader
        self.iter = iter(self.data_loader)

    def __next__(self):
        try:
            data = next(self.iter)
        except StopIteration:
            self.iter = iter(self.data_loader)
            data = next(self.iter)
        return data

    def __len__(self):
        return len(self.data_loader)


def class_counter(labels, num_class):
    class_weights = np.zeros(num_class)
    label_count = Counter(labels)
    for idx, (label, count) in enumerate(sorted(label_count.items(), key=lambda x: -x[1])):
        class_weights[label] = count
    return class_weights
