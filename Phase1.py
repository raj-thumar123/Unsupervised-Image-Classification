from transformers import ViTImageProcessor, ViTModel
#from PIL import Image
#import torch
from datasets import load_dataset

# Loading the Dataset 
# We are using the Cifar10 dataset, we can change it acoording to our need
dataset = load_dataset("cifar10")

# Taking the 100 images from Train split for the Feature extraction
images = dataset["train"]["img"][:100]  

#Loading the Vision Transformer Model & Vision Transformer Image Processor
processor = ViTImageProcessor.from_pretrained('facebook/dino-vitb16')
model = ViTModel.from_pretrained('facebook/dino-vitb16')

# Iterating over the images, extracting the features of images & printing the features of each image
for image_id, image in enumerate(images):

    # Processing each image
    inputs = processor(images=image, return_tensors="pt")
    outputs = model(**inputs)
    last_hidden_states = outputs.last_hidden_state
    
    # Print features of each image
    print(f"Features for image {image_id+1}:")
    print(last_hidden_states)
    print("Shape:", last_hidden_states.shape)
    print("-" * 70)

    
