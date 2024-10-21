import time

def measure_performance(dataset, num_samples=100):
    start_time = time.time()
    for i in range(num_samples):
        _ = dataset[i]
    end_time = time.time()
    return end_time - start_time