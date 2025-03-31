import os
import cv2
import numpy as np
from sklearn.model_selection import train_test_split
from tensorflow.keras.preprocessing.image import img_to_array
import matplotlib.pyplot as plt
from tensorflow.keras.models import Model, load_model
from tensorflow.keras.layers import Input, Conv2D, MaxPooling2D, Conv2DTranspose, Dropout, concatenate, BatchNormalization
from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping, ReduceLROnPlateau
from tensorflow.keras.optimizers import Adam
import tensorflow as tf

# Enable mixed precision for faster training if GPU supports it
try:
    import tensorflow as tf
    tf.keras.mixed_precision.set_global_policy('mixed_float16')
    print('Mixed precision enabled')
except:
    print('Mixed precision not supported, using default precision')

# Directories for your images and masks
image_dir = 'ISIC/Training data'  # Replace with the actual path to your images
mask_dir = 'ISIC/Training Ground Truth'    # Replace with the actual path to your masks

# Get the list of image and mask filenames
image_filenames = [f for f in os.listdir(image_dir) if f.endswith('.jpg')]
mask_filenames = [f for f in os.listdir(mask_dir) if f.endswith('_Segmentation.png')]

# Initialize lists to hold valid image-mask pairs
valid_images = []
valid_masks = []

# Loop through each image to match with its corresponding mask
for img_file in image_filenames:
    image_name = img_file.replace('.jpg', '')
    mask_file = f"{image_name}_Segmentation.png"
    if mask_file in mask_filenames:
        valid_images.append(os.path.join(image_dir, img_file))
        valid_masks.append(os.path.join(mask_dir, mask_file))

# Check the number of valid image-mask pairs
print(f"Number of valid image-mask pairs: {len(valid_images)}")

# Split the dataset into training and validation sets
X_train, X_val, y_train, y_val = train_test_split(valid_images, valid_masks, test_size=0.2, random_state=42)

# Function to load and preprocess images and masks with enhanced preprocessing
def load_image_and_mask(img_path, mask_path, img_size=(224, 224)):
    # Load image
    img = cv2.imread(img_path)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)  # Convert BGR to RGB
    
    # Load mask
    mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)
    
    # Resize both image and mask
    img = cv2.resize(img, img_size)
    mask = cv2.resize(mask, img_size)
    
    # Apply CLAHE for contrast enhancement
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    lab = cv2.cvtColor(img, cv2.COLOR_RGB2LAB)
    lab[:,:,0] = clahe.apply(lab[:,:,0])
    img = cv2.cvtColor(lab, cv2.COLOR_LAB2RGB)
    
    # Normalize
    img = img / 255.0
    mask = mask / 255.0
    
    # Convert to array
    img = img_to_array(img)
    mask = np.expand_dims(mask, axis=-1)
    
    return img, mask

# Process data in batches to reduce memory consumption
def batch_process_data(image_paths, mask_paths, img_size=(224, 224), batch_size=32):
    """Process images in batches to save memory"""
    num_samples = len(image_paths)
    num_batches = (num_samples + batch_size - 1) // batch_size
    
    all_images = []
    all_masks = []
    
    for i in range(num_batches):
        start_idx = i * batch_size
        end_idx = min((i + 1) * batch_size, num_samples)
        
        batch_img_paths = image_paths[start_idx:end_idx]
        batch_mask_paths = mask_paths[start_idx:end_idx]
        
        batch_images = []
        batch_masks = []
        
        for img_path, mask_path in zip(batch_img_paths, batch_mask_paths):
            img, mask = load_image_and_mask(img_path, mask_path, img_size)
            batch_images.append(img)
            batch_masks.append(mask)
        
        all_images.extend(batch_images)
        all_masks.extend(batch_masks)
        
        print(f"Processed batch {i+1}/{num_batches}")
    
    return np.array(all_images), np.array(all_masks)

# Set a smaller image size for faster training
IMG_SIZE = (224, 224)  # Reduced from 256x256 for faster training
BATCH_SIZE = 32

print("Processing training data...")
X_train_processed, y_train_processed = batch_process_data(X_train, y_train, IMG_SIZE, BATCH_SIZE)
print("Processing validation data...")
X_val_processed, y_val_processed = batch_process_data(X_val, y_val, IMG_SIZE, BATCH_SIZE)

# Data augmentation for training
from tensorflow.keras.preprocessing.image import ImageDataGenerator

# CORRECTED DATA GENERATOR FUNCTION
# CORRECTED DATA GENERATOR FUNCTION
def create_data_generator(batch_size=16):
    """
    Create a proper data generator that yields both images and masks
    """
    # Create generators with the same augmentation parameters
    data_gen_args = dict(
        rotation_range=15,
        width_shift_range=0.1,
        height_shift_range=0.1,
        zoom_range=0.1,
        horizontal_flip=True,
        vertical_flip=True,
        fill_mode='nearest'
    )
    
    image_datagen = ImageDataGenerator(**data_gen_args)
    mask_datagen = ImageDataGenerator(**data_gen_args)
    
    # Use the same seed to ensure corresponding augmentations
    seed = 42
    
    # Create a generator function that yields both images and masks
    def generator():
        image_gen = image_datagen.flow(
            X_train_processed, 
            batch_size=batch_size,
            seed=seed,
            shuffle=True
        )
        
        mask_gen = mask_datagen.flow(
            y_train_processed,
            batch_size=batch_size,
            seed=seed,
            shuffle=True
        )
        
        while True:
            # Replace image_gen.next() with next(image_gen)
            X_batch = next(image_gen)
            y_batch = next(mask_gen)
            yield X_batch, y_batch
    
    # Return the generator function and steps per epoch
    return generator(), len(X_train_processed) // batch_size
    
    image_datagen = ImageDataGenerator(**data_gen_args)
    mask_datagen = ImageDataGenerator(**data_gen_args)
    
    # Use the same seed to ensure corresponding augmentations
    seed = 42
    
    # Create a generator function that yields both images and masks
    def generator():
        image_gen = image_datagen.flow(
            X_train_processed, 
            batch_size=batch_size,
            seed=seed,
            shuffle=True
        )
        
        mask_gen = mask_datagen.flow(
            y_train_processed,
            batch_size=batch_size,
            seed=seed,
            shuffle=True
        )
        
        while True:
            X_batch = image_gen.next()
            y_batch = mask_gen.next()
            yield X_batch, y_batch
    
    # Return the generator function and steps per epoch
    return generator(), len(X_train_processed) // batch_size

# Calculate class weights to address imbalance
def compute_class_weights(y_data):
    """Compute class weights to handle imbalanced data"""
    neg_weight = np.sum(y_data == 0)
    pos_weight = np.sum(y_data > 0)
    total = neg_weight + pos_weight
    
    class_weight = {
        0: total / (2.0 * neg_weight),
        1: total / (2.0 * pos_weight)
    }
    return class_weight

class_weights = compute_class_weights(y_train_processed)
print(f"Class weights: {class_weights}")

# Define a custom Dice coefficient loss with type casting
def dice_coef(y_true, y_pred, smooth=1.0):
    # Cast both tensors to the same data type (float32)
    y_true_f = tf.cast(tf.reshape(y_true, [-1]), tf.float32)
    y_pred_f = tf.cast(tf.reshape(y_pred, [-1]), tf.float32)
    intersection = tf.reduce_sum(y_true_f * y_pred_f)
    return (2.0 * intersection + smooth) / (tf.reduce_sum(y_true_f) + tf.reduce_sum(y_pred_f) + smooth)

def dice_loss(y_true, y_pred):
    return 1.0 - dice_coef(y_true, y_pred)

# Combined loss function
def bce_dice_loss(y_true, y_pred):
    # Cast inputs to float32 for the binary crossentropy calculation as well
    y_true_32 = tf.cast(y_true, tf.float32)
    y_pred_32 = tf.cast(y_pred, tf.float32)
    return tf.keras.losses.binary_crossentropy(y_true_32, y_pred_32) + dice_loss(y_true, y_pred)
# Build an efficient segmentation model
def build_efficient_unet(input_shape=(224, 224, 3)):
    """Build a more efficient U-Net with fewer parameters for faster training"""
    inputs = Input(input_shape)

    # Encoder - reduced filter counts for faster training
    c1 = Conv2D(16, (3, 3), activation='relu', padding='same', kernel_initializer='he_normal')(inputs)
    c1 = BatchNormalization()(c1)
    c1 = Conv2D(16, (3, 3), activation='relu', padding='same', kernel_initializer='he_normal')(c1)
    c1 = BatchNormalization()(c1)
    p1 = MaxPooling2D((2, 2))(c1)
    
    c2 = Conv2D(32, (3, 3), activation='relu', padding='same', kernel_initializer='he_normal')(p1)
    c2 = BatchNormalization()(c2)
    c2 = Conv2D(32, (3, 3), activation='relu', padding='same', kernel_initializer='he_normal')(c2)
    c2 = BatchNormalization()(c2)
    p2 = MaxPooling2D((2, 2))(c2)
    
    c3 = Conv2D(64, (3, 3), activation='relu', padding='same', kernel_initializer='he_normal')(p2)
    c3 = BatchNormalization()(c3)
    c3 = Conv2D(64, (3, 3), activation='relu', padding='same', kernel_initializer='he_normal')(c3)
    c3 = BatchNormalization()(c3)
    p3 = MaxPooling2D((2, 2))(c3)
    
    c4 = Conv2D(128, (3, 3), activation='relu', padding='same', kernel_initializer='he_normal')(p3)
    c4 = BatchNormalization()(c4)
    c4 = Conv2D(128, (3, 3), activation='relu', padding='same', kernel_initializer='he_normal')(c4)
    c4 = BatchNormalization()(c4)
    p4 = MaxPooling2D((2, 2))(c4)
    
    # Bottleneck
    c5 = Conv2D(256, (3, 3), activation='relu', padding='same', kernel_initializer='he_normal')(p4)
    c5 = BatchNormalization()(c5)
    c5 = Conv2D(256, (3, 3), activation='relu', padding='same', kernel_initializer='he_normal')(c5)
    c5 = BatchNormalization()(c5)
    
    # Decoder
    u6 = Conv2DTranspose(128, (2, 2), strides=(2, 2), padding='same')(c5)
    u6 = concatenate([u6, c4])
    c6 = Conv2D(128, (3, 3), activation='relu', padding='same', kernel_initializer='he_normal')(u6)
    c6 = BatchNormalization()(c6)
    c6 = Conv2D(128, (3, 3), activation='relu', padding='same', kernel_initializer='he_normal')(c6)
    c6 = BatchNormalization()(c6)
    
    u7 = Conv2DTranspose(64, (2, 2), strides=(2, 2), padding='same')(c6)
    u7 = concatenate([u7, c3])
    c7 = Conv2D(64, (3, 3), activation='relu', padding='same', kernel_initializer='he_normal')(u7)
    c7 = BatchNormalization()(c7)
    c7 = Conv2D(64, (3, 3), activation='relu', padding='same', kernel_initializer='he_normal')(c7)
    c7 = BatchNormalization()(c7)
    
    u8 = Conv2DTranspose(32, (2, 2), strides=(2, 2), padding='same')(c7)
    u8 = concatenate([u8, c2])
    c8 = Conv2D(32, (3, 3), activation='relu', padding='same', kernel_initializer='he_normal')(u8)
    c8 = BatchNormalization()(c8)
    c8 = Conv2D(32, (3, 3), activation='relu', padding='same', kernel_initializer='he_normal')(c8)
    c8 = BatchNormalization()(c8)
    
    u9 = Conv2DTranspose(16, (2, 2), strides=(2, 2), padding='same')(c8)
    u9 = concatenate([u9, c1])
    c9 = Conv2D(16, (3, 3), activation='relu', padding='same', kernel_initializer='he_normal')(u9)
    c9 = BatchNormalization()(c9)
    c9 = Conv2D(16, (3, 3), activation='relu', padding='same', kernel_initializer='he_normal')(c9)
    c9 = BatchNormalization()(c9)
    
    outputs = Conv2D(1, (1, 1), activation='sigmoid')(c9)
    
    model = Model(inputs, outputs)
    model.compile(optimizer=Adam(learning_rate=3e-4), 
                 loss=bce_dice_loss,
                 metrics=['accuracy', dice_coef])
    return model

# Build model
print("Building model...")
model = build_efficient_unet(input_shape=(*IMG_SIZE, 3))
model.summary()

# Define callbacks for training
model_checkpoint_path = 'segmentation_model_best.h5'
checkpoint = ModelCheckpoint(
    model_checkpoint_path,
    monitor='val_dice_coef',
    mode='max',
    save_best_only=True,
    verbose=1
)

early_stopping = EarlyStopping(
    monitor='val_dice_coef',
    patience=7,
    mode='max',
    restore_best_weights=True,
    verbose=1
)

reduce_lr = ReduceLROnPlateau(
    monitor='val_dice_coef',
    factor=0.5,
    patience=3,
    mode='max',
    min_lr=1e-6,
    verbose=1
)

callbacks = [checkpoint, early_stopping, reduce_lr]

# Create data generator
train_generator, steps_per_epoch = create_data_generator(batch_size=BATCH_SIZE)

# FIXED TRAINING PART
print("Starting training...")
history = model.fit(
    train_generator,  # This is now a proper generator
    steps_per_epoch=steps_per_epoch,
    validation_data=(X_val_processed, y_val_processed),
    epochs=25,  # Reduced number of epochs
    callbacks=callbacks,
    verbose=1
)

# Save the final model
final_model_path = 'segmentation_model_final.h5'
model.save(final_model_path)
print(f"Final model saved to {final_model_path}")

# Load the best model from checkpoint
best_model = load_model(model_checkpoint_path, custom_objects={
    'dice_coef': dice_coef,
    'dice_loss': dice_loss,
    'bce_dice_loss': bce_dice_loss
})
print(f"Best model loaded from {model_checkpoint_path}")

# Save with custom name for easier reference
best_model_renamed = 'best_segmentation_model.h5'
best_model.save(best_model_renamed)
print(f"Best model saved with name {best_model_renamed}")

# Plot training history
def plot_training_history(history):
    """Plot training and validation metrics"""
    plt.figure(figsize=(12, 4))
    
    plt.subplot(1, 2, 1)
    plt.plot(history.history['loss'])
    plt.plot(history.history['val_loss'])
    plt.title('Model Loss')
    plt.ylabel('Loss')
    plt.xlabel('Epoch')
    plt.legend(['Train', 'Validation'], loc='upper right')
    
    plt.subplot(1, 2, 2)
    plt.plot(history.history['dice_coef'])
    plt.plot(history.history['val_dice_coef'])
    plt.title('Dice Coefficient')
    plt.ylabel('Dice Coefficient')
    plt.xlabel('Epoch')
    plt.legend(['Train', 'Validation'], loc='lower right')
    
    plt.tight_layout()
    plt.savefig('training_history.png')
    plt.show()

plot_training_history(history)

# Function to analyze infection intensity in images
def analyze_infection_intensity(image_path, model):
    """
    Analyze skin lesion image to determine infection percentage and intensity
    
    Args:
        image_path: Path to the image file
        model: Loaded segmentation model
        
    Returns:
        Dictionary containing analysis results
    """
    # Load and preprocess the image
    img = cv2.imread(image_path)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img_resized = cv2.resize(img, IMG_SIZE)
    
    # Apply CLAHE
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    lab = cv2.cvtColor(img_resized, cv2.COLOR_RGB2LAB)
    lab[:,:,0] = clahe.apply(lab[:,:,0])
    img_enhanced = cv2.cvtColor(lab, cv2.COLOR_LAB2RGB)
    
    # Normalize
    img_input = img_enhanced / 255.0
    img_input = np.expand_dims(img_input, axis=0)
    
    # Get segmentation prediction
    prediction = model.predict(img_input, verbose=0)[0]
    
    # Calculate infection percentage (percentage of pixels that are infected)
    # Using a threshold of 0.5 for binary segmentation
    binary_mask = prediction > 0.5
    infection_percentage = np.mean(binary_mask) * 100
    
    # Analyze intensity distribution within the infected region
    if np.sum(binary_mask) > 0:
        # Get the original image
        original_img = img_resized / 255.0
        
        # Calculate color channel intensities in the infected region
        red_intensity = np.mean(original_img[:,:,0][binary_mask[:,:,0]]) * 100
        green_intensity = np.mean(original_img[:,:,1][binary_mask[:,:,0]]) * 100
        blue_intensity = np.mean(original_img[:,:,2][binary_mask[:,:,0]]) * 100
        
        # Calculate HSV values for better color analysis
        hsv_img = cv2.cvtColor((original_img * 255).astype(np.uint8), cv2.COLOR_RGB2HSV)
        hsv_img = hsv_img / 255.0
        
        hue = np.mean(hsv_img[:,:,0][binary_mask[:,:,0]]) * 360  # Convert to 0-360 range
        saturation = np.mean(hsv_img[:,:,1][binary_mask[:,:,0]]) * 100
        value = np.mean(hsv_img[:,:,2][binary_mask[:,:,0]]) * 100
        
        # Grayscale intensity (simple average of RGB)
        grayscale_intensity = (red_intensity + green_intensity + blue_intensity) / 3
        
        # Calculate mean entropy in the lesion area (texture complexity)
        # Convert to grayscale
        gray_img = cv2.cvtColor((original_img * 255).astype(np.uint8), cv2.COLOR_RGB2GRAY)
        # Calculate local entropy
        entropy_kernel_size = 9
        entropy_img = np.zeros_like(gray_img, dtype=float)
        
        for i in range(entropy_kernel_size//2, gray_img.shape[0] - entropy_kernel_size//2):
            for j in range(entropy_kernel_size//2, gray_img.shape[1] - entropy_kernel_size//2):
                if binary_mask[i, j, 0]:
                    patch = gray_img[i-entropy_kernel_size//2:i+entropy_kernel_size//2+1, 
                                    j-entropy_kernel_size//2:j+entropy_kernel_size//2+1]
                    hist = cv2.calcHist([patch], [0], None, [256], [0, 256])
                    hist = hist / np.sum(hist)
                    entropy_img[i, j] = -np.sum(hist * np.log2(hist + 1e-7))
        
        mean_entropy = np.mean(entropy_img[binary_mask[:,:,0]])
        
        # Create intensity score (0-100)
        # Lower grayscale intensity (darker lesions) often indicate higher severity
        intensity_score = 100 - grayscale_intensity
        
        # Create texture score (0-100)
        # Higher entropy (more complex texture) often indicates higher severity
        # Normalize entropy to 0-100 scale (typical entropy values range from 0-8)
        texture_score = min(100, mean_entropy * 12.5)
        
        # Create a severity score based on multiple factors
        # This formula can be tuned based on domain expertise
        severity_score = (0.4 * infection_percentage) + (0.3 * intensity_score) + (0.3 * texture_score)
        
        # Classify severity
        if severity_score < 30:
            severity = "Mild"
        elif severity_score < 60:
            severity = "Moderate"
        else:
            severity = "Severe"
            
        # Create a heatmap overlay to visualize infection intensity
        heatmap = np.zeros_like(img_resized)
        
        # Create RGB heatmap based on prediction confidence
        # Red channel shows infection intensity
        heatmap[:,:,0] = (prediction[:,:,0] * 255).astype(np.uint8)  
        # Green and blue channels show healthier areas
        heatmap[:,:,1] = ((1 - prediction[:,:,0]) * 255).astype(np.uint8)
        heatmap[:,:,2] = ((1 - prediction[:,:,0]) * 200).astype(np.uint8)
        
        # Blend with original image
        alpha = 0.6
        blended_img = cv2.addWeighted(img_resized, 1-alpha, heatmap, alpha, 0)
        
        # Return comprehensive analysis
        return {
            'infection_percentage': infection_percentage,
            'red_intensity': red_intensity,
            'green_intensity': green_intensity,
            'blue_intensity': blue_intensity,
            'hue': hue,
            'saturation': saturation,
            'value': value,
            'grayscale_intensity': grayscale_intensity,
            'texture_complexity': mean_entropy,
            'intensity_score': intensity_score,
            'texture_score': texture_score,
            'severity_score': severity_score,
            'severity_classification': severity,
            'original_image': img_resized,
            'segmentation_mask': prediction,
            'binary_mask': binary_mask,
            'visualization': blended_img
        }
    else:
        # Return limited analysis for non-infected images
        return {
            'infection_percentage': 0,
            'severity_classification': 'None',
            'severity_score': 0,
            'original_image': img_resized,
            'segmentation_mask': prediction,
            'binary_mask': binary_mask,
            'visualization': img_resized
        }

# Function to display analysis results
def display_analysis_results(results):
    """Display visual results of infection analysis"""
    plt.figure(figsize=(18, 10))
    
    # Original image
    plt.subplot(2, 3, 1)
    plt.imshow(results['original_image'])
    plt.title('Original Image')
    plt.axis('off')
    
    # Segmentation mask (probability map)
    plt.subplot(2, 3, 2)
    plt.imshow(results['segmentation_mask'], cmap='jet')
    plt.title(f'Segmentation Probability')
    plt.axis('off')
    plt.colorbar(fraction=0.046, pad=0.04)
    
    # Binary mask
    plt.subplot(2, 3, 3)
    plt.imshow(results['binary_mask'][:,:,0], cmap='gray')
    plt.title(f'Binary Mask\nInfection: {results["infection_percentage"]:.2f}%')
    plt.axis('off')
    
    # Visualization with heatmap
    plt.subplot(2, 3, 4)
    plt.imshow(results['visualization'])
    plt.title(f'Severity: {results["severity_classification"]}')
    plt.axis('off')
    
    # Add detailed metrics as text
    if 'severity_score' in results:
        metrics_text = f"Severity Score: {results['severity_score']:.2f}\n"
        if 'grayscale_intensity' in results:
            metrics_text += (
                f"RGB Intensity: R={results['red_intensity']:.1f}, "
                f"G={results['green_intensity']:.1f}, "
                f"B={results['blue_intensity']:.1f}\n"
                f"Grayscale Intensity: {results['grayscale_intensity']:.1f}\n"
            )
        if 'texture_complexity' in results:
            metrics_text += f"Texture Complexity: {results['texture_complexity']:.2f}\n"
            
        plt.subplot(2, 3, 5)
        plt.axis('off')
        plt.text(0.1, 0.5, metrics_text, fontsize=12, va='center')
        plt.title('Detailed Metrics')
    
    plt.tight_layout()
    plt.savefig('analysis_results.png')
    plt.show()

# Example of using the analysis function
def analyze_sample_images(model, image_dir, num_samples=3):
    """
    Analyze a sample of images from the dataset
    
    Args:
        model: Trained segmentation model
        image_dir: Directory containing images
        num_samples: Number of images to analyze
    """
    # Get a sample of image files
    image_files = [f for f in os.listdir(image_dir) if f.endswith('.jpg')]
    if num_samples > len(image_files):
        num_samples = len(image_files)
    
    sample_files = np.random.choice(image_files, num_samples, replace=False)
    
    for i, file in enumerate(sample_files):
        image_path = os.path.join(image_dir, file)
        print(f"\nAnalyzing image {i+1}/{num_samples}: {file}")
        
        # Analyze the image
        results = analyze_infection_intensity(image_path, model)
        
        # Display results
        display_analysis_results(results)
        
        # Print detailed analysis
        print(f"Infection Percentage: {results['infection_percentage']:.2f}%")
        print(f"Severity Classification: {results['severity_classification']}")
        print(f"Severity Score: {results.get('severity_score', 0):.2f}")
        
        if 'grayscale_intensity' in results:
            print(f"Intensity Analysis:")
            print(f"  - Red Channel: {results['red_intensity']:.2f}%")
            print(f"  - Green Channel: {results['green_intensity']:.2f}%")
            print(f"  - Blue Channel: {results['blue_intensity']:.2f}%")
            print(f"  - Grayscale: {results['grayscale_intensity']:.2f}%")
        
        if 'texture_complexity' in results:
            print(f"Texture Complexity: {results['texture_complexity']:.2f}")
            print(f"Texture Score: {results['texture_score']:.2f}")

# Usage examples - uncomment to run after training
# 1. Analyze a single image
# results = analyze_infection_intensity('path/to/your/image.jpg', best_model)
# display_analysis_results(results)

# 2. Analyze a batch of sample images
# analyze_sample_images(best_model, image_dir, num_samples=3)

print("\nTraining and model setup complete! You can now use the analyze_infection_intensity() function to process new images.")
print("The best model is saved as 'best_segmentation_model.h5' and the final model as 'segmentation_model_final.h5'")