
import sys
import datasets
import huggingface_hub
import cv2
import pandas

print("Setup check successful!")
print("Python version:", sys.version.split()[0])
print("Datasets version:", datasets.__version__)
print("Hugging Face Hub version:", huggingface_hub.__version__)
print("OpenCV version:", cv2.__version__)
print("Pandas version:", pandas.__version__)
