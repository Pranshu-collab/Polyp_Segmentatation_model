# Polyp_Segmentatation_model
## Here I have implemented a neural network for segmentation of gastrointestinal polyps (which are precursors to colorectal cancer) in colonoscopy images
### Loading the Input dataset 
I have mounted the drive on colab and then loaded the images using load_polyp_dataset which matches the image file with its corresponding mask and stored it as an array of its features . Also the pixels of RGB are normalized between 0 and 1 .
