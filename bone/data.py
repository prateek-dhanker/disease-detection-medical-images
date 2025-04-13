import kagglehub

# Download latest version
path = kagglehub.dataset_download("devbatrax/fracture-detection-using-x-ray-images")

print("Path to dataset files:", path)