# @title ## 3. download LoRA and test files
# @markdown (If you meet problems of downloading the two files using the following code, you can download them manually according to their links and then put them into their respective folders.)
#!pip install wget
import wget

# URL of the pre-trained LoRA file on Google Drive
lora_url = "https://drive.google.com/u/0/uc?id=1MiaG8v0ZmkTwwrzIEFtVoBj-Jjqi_5lz&export=download"

# Destination path to save the downloaded file
lora_save_path = "./lora.safetensors"

# Download the file using wget
wget.download(lora_url, lora_save_path)

# # download test file
# !wget -O /content/kohya-trainer/stitchdiffusion_test.py https://raw.githubusercontent.com/lshus/stitchdiffusion-colab/main/stitchdiffusion_test.py