# Polyp_Segmentatation_model
## Here I have implemented a neural network for segmentation of gastrointestinal polyps (which are precursors to colorectal cancer) in colonoscopy images

### Loading and Preprocessing the Dataset
For this task, we used the Kvasir-SEG dataset, which contains colonoscopy images along with corresponding polyp segmentation masks. The dataset can be downloaded from Kvasir-SEG official site.
We mounted Google Drive in Colab to access the dataset files. The dataset loading process involves the following steps:
#### Loading Images and Masks:
We implemented a function load_polyp_dataset that reads images and their corresponding masks from specified directories. This function ensures that each image is matched correctly with its mask by sorting the file names and pairing them accordingly.
#### Preprocessing:
Each image and mask is resized to a uniform size (256x256 pixels). Images are normalized by scaling pixel values to the range [0,1]. Masks are converted to binary masks by thresholding grayscale values to separate polyp areas from the background. Masks are reshaped to add a channel dimension, matching model input requirements.
#### Dataset Splitting and Tensor Conversion:
We created a function get_tf_dataset which splits the loaded data into training, validation, and test sets. It also converts the NumPy arrays into TensorFlow tensors, preparing the data pipeline for efficient processing.
#### Batching and Prefetching:
To improve training efficiency, the datasets are batched and prefetched. This allows the model to receive data in batches and prepares the next batch in advance during training, reducing bottlenecks.
#### Data Augmentation:
To increase data variability and help prevent overfitting, we implemented an augmentation function called augment. This function applies random horizontal flipping to both images and their masks simultaneously, preserving alignment.
#### Summary:
After running these functions, the images and masks are loaded as NumPy arrays, preprocessed, and split into train_ds, val_ds, and test_ds TensorFlow datasets. These are ready for training and evaluation of the segmentation model.

### Implementation of Attention with UNET Architecture 
For the polyp segmentation task, we designed and implemented an Attention U-Net model, an enhanced version of the classic U-Net architecture. The Attention U-Net incorporates attention gates to improve the model’s focus on relevant regions of the input image (i.e., the polyps), which are often small and vary greatly in shape and size.

#### Architecture Details:
Encoder-Decoder Backbone:
* The model follows the well-known U-Net pattern with an encoder (downsampling path) and a decoder (upsampling path).
* The encoder extracts hierarchical feature representations at multiple spatial scales via convolutional blocks and max pooling.
* The decoder progressively upsamples these features to reconstruct the spatial resolution and generate pixel-wise segmentation masks.

#### Convolutional Blocks:
Each block consists of two convolutional layers with ReLU activations and batch normalization to stabilize and speed up training.

#### Attention Gates:
Attention gates are integrated into the skip connections between encoder and decoder. They learn to suppress irrelevant background regions in encoder feature maps before concatenation, allowing the model to focus more precisely on polyp regions. This attention mechanism helps improve segmentation accuracy, especially for small and complex lesions.

#### Number of Filters:
The network starts with 16 filters in the first convolutional layer and doubles the number of filters after each downsampling step, capturing increasingly complex features.

#### Output Layer:
A 1x1 convolution followed by a sigmoid activation produces the final per-pixel probability mask for polyp presence.

##### Why Attention U-Net?
* Handling Small and Variable Lesions: Polyps are often small and can be obscured by noise or complex backgrounds. Attention gates help the model to dynamically weigh the importance of features spatially, enhancing segmentation precision.
* Efficient Skip Connections: Unlike traditional U-Net that blindly concatenates encoder features, attention gates modulate these features, reducing irrelevant information and making learning more focused.
* Improved Interpretability: Attention maps can be visualized to understand which parts of the image influenced the model’s decision, aiding clinical trust.

### Model Architecture and Loss Functions

#### Model Architecture:
We implemented an Attention U-Net for binary polyp segmentation. This architecture extends the classic U-Net by integrating attention gates in the skip connections. These gates help the model focus on relevant spatial regions, improving segmentation accuracy by suppressing irrelevant background features.

#### Input Shape and Filters:
The input to the model is an RGB image resized to 256×256×3. We start with 32 filters in the first convolutional layer, doubling filters at each downsampling step, which balances model capacity and computational efficiency.

#### Output:
Since this is a binary segmentation problem (polyp vs. background), the output layer uses a sigmoid activation with a single filter (channel), predicting the probability of polyp presence at each pixel.

#### Loss Functions:
We combined Binary Cross-Entropy (BCE) and Dice Loss to form a composite loss function:
* Binary Cross-Entropy (BCE): Measures pixel-wise classification error.
* Dice Loss: Measures overlap between predicted masks and ground truth, directly optimizing segmentation quality, especially helpful for imbalanced classes where polyp regions may be small.
The combined loss is BCE + 0.5 * Dice Loss, which balances pixel-wise accuracy with region-level overlap.

#### Evaluation Metrics:
We track Dice Coefficient and Intersection over Union (IoU) as key segmentation performance metrics, which better reflect the quality of mask predictions than accuracy alone.

### Model Compilation
The model is compiled using the Adam optimizer, which is well-suited for segmentation tasks due to its adaptive learning rate capabilities and fast convergence.
The loss function used is a combination of Binary Crossentropy (BCE) and Dice Loss. This hybrid loss leverages both pixel-level classification accuracy (via BCE) and region-level overlap accuracy (via Dice Loss), making it particularly effective for medical segmentation tasks like polyp detection, where foreground-background imbalance is common.
We use two performance metrics:
* Dice Coefficient: Measures the harmonic mean of precision and recall to evaluate overlap.
* Intersection over Union (IoU): Measures the intersection divided by the union of predicted and ground truth masks.
Additionally, model.summary() is used to print a detailed overview of the architecture, and plot_model() (from keras.utils) is called to visualize the entire Attention U-Net structure, including encoder-decoder paths and attention gates.

### Model Training
To ensure optimal training performance and efficient resource usage, the following training strategies were employed:

#### Model Checkpointing:
A ModelCheckpoint callback was used to save the model with the lowest validation loss during training. This helps preserve the best-performing model for later evaluation or deployment.

#### Early Stopping:
The training process was equipped with EarlyStopping to halt training automatically when the validation performance stops improving for a defined number of epochs (patience). This prevents overfitting and unnecessary computation.

#### ReduceLROnPlateau:
To enhance convergence, the learning rate was automatically reduced when the validation loss plateaued. This allows the model to make finer adjustments during later stages of training.

#### Epochs:
The model was trained for up to 50 epochs, with the callbacks ensuring that the process stops early if necessary.

#### Validation 
Validation loss , Validation dice coefficient and Validation iou coeffcient are computed from the model.

### Validation Visualization 
#### Extracting Validation Data
The validation features and masks, X_val and y_val, were extracted directly from the val_ds TensorFlow dataset by iterating over its batches and collecting a manageable number of samples for visualization.

#### Visualizing Sample Predictions
To evaluate the model’s segmentation performance qualitatively, we selected 5 random samples from the validation set and performed inference using the trained model. For each sample, the following were visualized:
* The original image
* The corresponding ground truth mask
* The predicted mask output from the model
These visualizations help assess how accurately the model identifies and segments the polyp regions in unseen data. The visual comparison provides intuitive insights into model performance, highlighting strengths and any areas where the model may need improvement.

### Evaluation of the model on testing set
After training, the model was evaluated on the testing set (test_ds) to assess its generalization capability on completely unseen data. The evaluation was performed using the following metrics:
* Binary Crossentropy + Dice Loss (used during training)
* Dice Coefficient: Measures the overlap between the predicted and ground truth masks. A higher value indicates better segmentation.
* Intersection over Union (IoU) Coefficient: Quantifies the accuracy of predicted segmentation masks by computing the overlap divided by the union of predicted and ground truth areas.
