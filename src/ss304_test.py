from ss304_dataset import ss304Dataset

test_dataset = ss304Dataset(data_type='test')
print(test_dataset[3000])
test_dataset.check_image(3000)
