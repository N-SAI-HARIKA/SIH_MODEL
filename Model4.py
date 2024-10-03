import tensorflow as tf
import pandas as pd
import numpy as np
import os
from keras.models import Model
from keras.layers import Dense, GlobalAveragePooling2D, BatchNormalization
from keras.optimizers import Adam
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from keras.preprocessing.image import ImageDataGenerator
from keras.callbacks import LearningRateScheduler, EarlyStopping

# Assuming target size for InceptionV3 is (299, 299)
target_size = (299, 299)

def load_and_preprocess_image(file_path, target_size=target_size):
    image = tf.keras.preprocessing.image.load_img(file_path, target_size=target_size)
    image_array = tf.keras.preprocessing.image.img_to_array(image)
    image_array = tf.keras.applications.inception_v3.preprocess_input(image_array)
    return image_array

def create_model(target_size):
    base_model = tf.keras.applications.InceptionV3(weights='imagenet', include_top=False, input_shape=(target_size[0], target_size[1], 3))
    
    for layer in base_model.layers:
        layer.trainable = False

    x = base_model.output
    x = GlobalAveragePooling2D()(x)
    x = Dense(128, activation='relu')(x)
    x = BatchNormalization()(x)
    x = tf.keras.layers.Dropout(0.5)(x)
    predictions = Dense(1, activation='sigmoid')(x)

    model = Model(inputs=base_model.input, outputs=predictions)
    model.compile(optimizer=Adam(), loss='binary_crossentropy', metrics=['accuracy'])
    
    return model

def train_model(model, train_generator, val_generator, epochs=10):
    lr_schedule = LearningRateScheduler(lambda epoch: 1e-3 if epoch < 5 else (1e-4 if 5 <= epoch < 10 else 1e-5))
    early_stopping = EarlyStopping(monitor='val_loss', patience=3, restore_best_weights=True)
    
    model.fit(train_generator, epochs=epochs, validation_data=val_generator, callbacks=[lr_schedule, early_stopping])

def evaluate_model(model, test_generator):
    results = model.evaluate(test_generator)
    print("Test Loss:", results[0])
    print("Test Accuracy:", results[1])

# Paths for clean and dirt images
clean_folder = "Clean"
dirt_folder = "Dirt Buildup"
# Get a list of file paths for clean and dirt images
clean_files = [os.path.join(clean_folder, file) for file in os.listdir(clean_folder)]
dirt_files = [os.path.join(dirt_folder, file) for file in os.listdir(dirt_folder)]

# Create a DataFrame with file paths and labels (0 for clean, 1 for dirt)
data = {'file_path': clean_files + dirt_files, 'label': [0] * len(clean_files) + [1] * len(dirt_files)}
df = pd.DataFrame(data)

# Optionally, you can shuffle the DataFrame
df = df.sample(frac=1).reset_index(drop=True)

# Display the DataFrame information
print("DataFrame Information:")
print(df.info())

# Load and preprocess images
df['image'] = df['file_path'].apply(load_and_preprocess_image)

# Standardize pixel values
scaler = StandardScaler()

# Resize and load images into NumPy array
images = np.array([np.array(image) for image in df['image']])

# Standardize pixel values directly
images_standardized = scaler.fit_transform(images.reshape(images.shape[0], -1))

# Ensure that the standardized array has a consistent shape
images_standardized = np.vstack(images_standardized).reshape(images.shape)

print(len(df['file_path']))
print(len(df['label']))
print(len(images_standardized))

# Output the DataFrame with standardized pixel values
df_final = pd.DataFrame({'file_path': df['file_path'], 'label': df['label'].astype(str), 'image': images_standardized.tolist()})
print("\nDataFrame with Standardized Pixel Values:")
print(df_final.head())

# Get a list of file paths for clean and dirt images
clean_files = [os.path.join(clean_folder, file) for file in df_final['file_path']]
dirt_files = [os.path.join(dirt_folder, file) for file in df_final['file_path']]

# Create a DataFrame with file paths and labels
# Pad all lists in the 'image' column with zeros
max_length = max(len(image) for image in df_final['image'])
df_final['image'] = df_final['image'].apply(lambda x: x + [0] * (max_length - len(x)))


# Split the dataset into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(df_final['file_path'], df_final['label'], test_size=0.2, random_state=42)

# Create an InceptionV3 base model
model = create_model(target_size)

# Create data generators with data augmentation for training
train_datagen = ImageDataGenerator(rescale=1./255, rotation_range=20, width_shift_range=0.2, height_shift_range=0.2, shear_range=0.2, zoom_range=0.2, horizontal_flip=True, fill_mode='nearest')

# No data augmentation for validation/testing
test_datagen = ImageDataGenerator(rescale=1./255)
batch_size = 32
# Define the validation generator
val_generator = test_datagen.flow_from_dataframe(dataframe=df_final, directory=None, x_col='file_path', y_col='label', target_size=target_size, batch_size=batch_size, class_mode='binary', shuffle=False)

# Train the model
batch_size = 32
train_generator = train_datagen.flow_from_dataframe(dataframe=df_final, directory=None, x_col='file_path', y_col='label', target_size=target_size, batch_size=batch_size, class_mode='binary', shuffle=True)

# Evaluate the model on the test set
test_generator = test_datagen.flow_from_dataframe(dataframe=df_final,directory=None,x_col='file_path',y_col='label',target_size=target_size,batch_size=batch_size,class_mode='binary',shuffle=False)
# Train and evaluate the model
train_model(model, train_generator, val_generator, epochs=10)
evaluate_model(model, test_generator)

# Load the trained model from the HDF5 file
model = tf.keras.models.load_model('model4.h5')

from keras.preprocessing import image
from IPython.display import display
import ipywidgets as widgets
from IPython.display import display as ipydisplay

# Function to preprocess user-input image
def preprocess_user_image(img, target_size=target_size):
    img_array = image.img_to_array(img)
    img_array = np.expand_dims(img_array, axis=0)
    img_array = tf.keras.applications.inception_v3.preprocess_input(img_array)
    return img_array

# Function to predict the class of user-input image
def predict_user_image(model, img):
    preprocessed_img = preprocess_user_image(img)
    prediction = model.predict(preprocessed_img)
    return prediction

# Function to handle image upload and prediction
def handle_image_upload(change):
    img = change['new']
    
    # Display the user-input image
    display(img)
    
    # Make prediction using the trained model
    prediction_result = predict_user_image(model, img)

    # Display the prediction result
    if prediction_result[0][0] > 0.5:
        prediction_label = "Dirt Buildup"
    else:
        prediction_label = "Clean"

    # Display the prediction result and ask for user input
    user_feedback = input(f"Prediction: {prediction_label}\nIs the prediction correct? (yes/no): ")

    # Process user feedback
    if user_feedback.lower() == "yes":
        print("Great! Thank you for confirming.")
    else:
        print("Apologies for the incorrect prediction. We appreciate your feedback.")

# Create a file upload widget
file_upload = widgets.FileUpload(accept='image/*', multiple=False)
file_upload.observe(handle_image_upload, names='value')

# Display the file upload widget
ipydisplay(file_upload)

# Save the model to an HDF5 file
model.save('model4.h5')
