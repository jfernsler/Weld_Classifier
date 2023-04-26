from ss304_dataset import ss304Dataset
import random

def check_set(type = 'test'):
    test_dataset = ss304Dataset(data_type=type)
    rand_idx = random.randint(0, len(test_dataset))
    test_dataset.check_image(rand_idx)

check_set()