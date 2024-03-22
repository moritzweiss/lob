import ray

# Initialize Ray. This automatically detects the number of available CPUs
ray.init()

# Get a dictionary of all available resources
resources = ray.cluster_resources()

# The number of CPU cores is under the key "CPU"
num_cpus = resources.get("CPU", 0)

print(f"Number of CPU cores available to Ray: {num_cpus}")
