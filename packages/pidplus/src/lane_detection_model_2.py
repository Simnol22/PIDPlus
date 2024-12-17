import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
from PIL import Image
import os
import torch.optim as optim
from torchvision import transforms,models
from torchvision.transforms.functional import crop
from torch.utils.data import Dataset, DataLoader, random_split
import numpy as np
import cv2
from tqdm import tqdm



class LaneDetectionCNN(nn.Module):
    def __init__(self, input_shape=(3, 224, 224)):
        super(LaneDetectionCNN, self).__init__()
        self.conv1 = nn.Conv2d(3, 16, kernel_size=5, stride=2, padding=2)
        self.conv2 = nn.Conv2d(16, 32, kernel_size=5, stride=1, padding=2)
        self.maxpool1 = nn.MaxPool2d(kernel_size=2, stride=2)
        self.conv3 = nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1)
        self.maxpool2 = nn.MaxPool2d(kernel_size=2, stride=2)
        self.conv4 = nn.Conv2d(64, 128, kernel_size=3, stride=1, padding=1)
        self.maxpool3 = nn.MaxPool2d(kernel_size=2, stride=2)
        self.conv5 = nn.Conv2d(128, 256, kernel_size=3, stride=1, padding=1)
        self.maxpool4 = nn.MaxPool2d(kernel_size=2, stride=2)
        self.dropout = nn.Dropout(0.5)

        # Calculate flat size dynamically
        self._to_linear = None
        self._calculate_flat_size(input_shape)

        self.fc1 = nn.Linear(self._to_linear, 7*7*3) # 7x7 image with 3 channels
        self.fc2 = nn.Linear(7*7*3, 1)  # Single output neuron for regression

    def _calculate_flat_size(self, input_shape):
        x = torch.zeros(1, *input_shape)
        x = self._forward_conv(x)
        self._to_linear = x.numel()

    def _forward_conv(self, x):
        x = torch.relu(self.conv1(x))
        x = torch.relu(self.conv2(x))
        x = self.maxpool1(x)
        x = torch.relu(self.conv3(x))
        x = self.maxpool2(x)
        x = torch.relu(self.conv4(x))
        x = self.maxpool3(x)
        x = torch.relu(self.conv5(x))
        x = self.maxpool4(x)
        x = self.dropout(x)
        return x

    def forward(self, x):
        x = self._forward_conv(x)
        x = x.view(x.size(0), -1) 
        x = torch.relu(self.fc1(x))
        x = torch.tanh(self.fc2(x))
        x = x*0.5 #Scale output to be between -0.5 and 0.5
        return x
    


def apply_preprocessing_2(image):
    """
    Apply preprocessing transformations to the input image.

    Parameters:
    - image: PIL Image object.
    """

    image_array = np.array(image)

    blurred_image_array = cv2.GaussianBlur(image_array, (0, 0), 0.1)
    channels = [image_array[:, :, i] for i in range(image_array.shape[2])]
    h, w, _ = image_array.shape
    
    imghsv = cv2.cvtColor(blurred_image_array, cv2.COLOR_RGB2HSV)
    img = cv2.cvtColor(image_array, cv2.COLOR_RGB2GRAY)

    mask_ground = np.ones(img.shape, dtype=np.uint8)  # Start with a mask of ones (white)


    one_third_height = h // 3
    # crop_height = h * 2 // 5 
    mask_ground[:one_third_height, :] = 0  # Mask the top 1/3 of the image
    
    #gaussian filter
    sigma = 4.5
    img_gaussian_filter = cv2.GaussianBlur(img,(0,0), sigma)
    
    sobelx = cv2.Sobel(img_gaussian_filter,cv2.CV_64F,1,0)
    sobely = cv2.Sobel(img_gaussian_filter,cv2.CV_64F,0,1)
    # Compute the magnitude of the gradients
    Gmag = np.sqrt(sobelx*sobelx + sobely*sobely)
    threshold = 51


    white_lower_hsv = np.array([0, 0, 143])         # CHANGE ME
    white_upper_hsv = np.array([228, 60, 255])   # CHANGE ME
    yellow_lower_hsv = np.array([10, 50, 100])        # CHANGE ME
    yellow_upper_hsv = np.array([70, 255, 255])  # CHANGE ME

    mask_white = cv2.inRange(imghsv, white_lower_hsv, white_upper_hsv)
    mask_yellow = cv2.inRange(imghsv, yellow_lower_hsv, yellow_upper_hsv)

    # crop two fifth of the image on the right for the yello mask
    height, width = mask_yellow.shape 
    crop_width = width * 2 // 5 
    crop_width_2 = width * 1 // 2 
    crop_mask_11 = np.zeros_like(mask_yellow, dtype=np.uint8)
    crop_mask_11[:, :width - crop_width_2] = 1 
    mask_yellow = mask_yellow * crop_mask_11

    # crop two fifth of the image on the left for the white mask
    crop_mask_22 = np.zeros_like(mask_white, dtype=np.uint8)
    crop_mask_22[:, crop_width:] = 1 
    mask_white = mask_white * crop_mask_22


    mask_mag = (Gmag > threshold)

    # np.savetxt("mask.txt", mask_white, fmt='%d', delimiter=',')
    # exit()
    crop_width_3 = width * 1 // 10 
    crop_mask_33 = np.zeros_like(mask_yellow, dtype=np.uint8)
    crop_mask_33[:, :width - crop_width_3] = 1 

    crop_mask_44 = np.zeros_like(mask_white, dtype=np.uint8)
    crop_mask_44[:, crop_width_3:] = 1 

    final_mask = mask_ground * mask_mag * 255 
    mask_white = mask_ground * mask_white
    mask_yellow = mask_ground * mask_yellow
    # Convert the NumPy array back to a PIL image

    channels[0] =  final_mask #np.zeros_like(channels[0])
    channels[1] =  mask_white
    channels[2] =  mask_yellow
    
    filtered_image = np.stack(channels, axis=-1)
    filtered_image = Image.fromarray(filtered_image)
    return  filtered_image


# display image after filter application
# # image = Image.open('example.png') 
# ii=0
# while True:
#     image = cv2.imread(f"lab_images/image_{ii}.jpg")

#     print(image.shape)

#     # create an transform for crop top 1/3 of the image
#     transform = transforms.Compose([
#         transforms.Lambda(apply_preprocessing)
#     ])
    
#     image_new = transform(image) 
    
#     # display(image)
#     # display(image_crop)
#     image_new = np.array(image_new)
#     cv2.imshow("Image", image_new)
#     cv2.imshow("Imageee", image)
#     print(f"image_{ii}")
#     cv2.waitKey(0)
#     ii+=1



# Dataset class
# class SequentialImageDataset(Dataset):
#     def __init__(self, image_folder, label_folder, transform=None):
#         self.image_folder = image_folder
#         self.label_folder = label_folder
#         self.transform = transform

#         self.image_files = sorted(os.listdir(image_folder))
#         self.label_files = sorted(os.listdir(label_folder))

#     def __len__(self):
#         return len(self.image_files)

#     def __getitem__(self, idx):
#         img_path = os.path.join(self.image_folder, self.image_files[idx])
#         lbl_path = os.path.join(self.label_folder, self.label_files[idx])

#         # Load and preprocess image
#         image = Image.open(img_path).convert("RGB")
#         if self.transform:
#             image = self.transform(image)

#         # Load label
#         with open(lbl_path, "r") as f:
#             label = float(f.read().strip())

#         return image, torch.tensor(label, dtype=torch.float32)

class SequentialImageDataset(Dataset):
    def __init__(self, data_1=None, data_2=None, data_3=None, transform=None):
        self.data_1 = data_1
        self.data_2 = data_2
        self.data_3 = data_3
        self.transform = transform

        # Load all image paths from the specified datasets
        self.images = []
        self.labels = []

        if data_1 is not None:
            self._load_data(data_1)

        if data_2 is not None:
            self._load_data(data_2)

        if data_3 is not None:
            self._load_data(data_3)

    def _load_data(self, data_path):
        images_path = os.path.join(data_path, "images")
        labels_path = os.path.join(data_path, "labels")

        image_files = os.listdir(images_path)  # List all image files
        for i in range(len(image_files)):      # Iterate through indices
            if image_files[i].endswith(".png"):  # Ensure the file is an image
                # Construct image path
                self.images.append(os.path.join(images_path, image_files[i]))
                # Construct label path using the index `i`
                self.labels.append(os.path.join(labels_path, f"{i}.txt"))  # Use the index to find the corresponding label


    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        # Load the image
        image = Image.open(self.images[idx]).convert("RGB")  # Ensure RGB mode

        # Load the label
        label_path = self.labels[idx]
        with open(label_path, "r") as f:
            label = float(f.read().strip())

        # Apply transformations if specified
        if self.transform:
            image = self.transform(image)

        # Return image and label as a tensor
        return image, torch.tensor(label, dtype=torch.float32)


def get_transform():
    transform = transforms.Compose([
        transforms.Lambda(apply_preprocessing),
        transforms.ToTensor(),  # Convert image to tensor
    ])
    return transform

# the following functions are to be used when using a trained model for real time predictions:
def predict_dist(model, im, device):
    transform = get_transform()
    im = transform(im)
    im = im.unsqueeze(0).to(device)

    # Forward pass through the model
    with torch.no_grad():
        prediction = model(im)


    # Extract the scalar prediction
    predicted_distance = prediction.item()

    return predicted_distance

def load_CNN_model(model_path, input_shape, device, rnn_hidden_size=128):
    model = LaneDetectionCNN(input_shape)
    # Load state dictionary
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.to(device)
    model.eval()
    print(f"CNN model loaded successfully from '{model_path}'")
    return model

def show_image(img):
    transform = get_transform()
    img = transform(img)
    img = img.permute(1, 2, 0).cpu().numpy()
    cv2.imshow("Image", img)
    # cv2.imshow("Imageee", im_pp)
    # im = np.array(Image.fromarray(obs))
    cv2.waitKey(1)




if __name__ == "__main__":

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # Dataset paths
    DATA_1 = None #"training_images/trail1"
    DATA_2 = None#"training_images/trail2"
    DATA_3 = "training_images/trail3"


    LOAD_MODEL = None#"models/lane_detection_cnn5.pth"

    batch_size = 64
    n_epochs = 10
    learning_rate = 0.001
    train_fraction = 0.8
    val_fraction = 0.1
    test_fraction = 0.1

    # Create DataLoaders
    train_loader, val_loader, test_loader = get_sequential_dataloader(
        data_1=DATA_1, data_2=DATA_2, data_3=DATA_3, 
        batch_size= batch_size, 
        train_fraction=train_fraction, val_fraction=val_fraction, test_fraction=test_fraction
    )


    # Initialize model, loss function, and optimizer
    input_shape = (3, 480, 640)  # Update to reflect 480x640 image size
    if LOAD_MODEL == None:
        model = LaneDetectionCNN(input_shape)
        criterion = nn.MSELoss()
        optimizer = optim.Adam(model.parameters(), lr=learning_rate)
    else:
        model = load_CNN_model(LOAD_MODEL, input_shape, device)
        print(f"model loaded from {LOAD_MODEL}")
        print("testing loaded model performance...")
        criterion = nn.MSELoss()
        optimizer = optim.Adam(model.parameters(), lr=learning_rate)
        avg_val_loss = test_cnn_model(model, val_loader, criterion, device)
        print(f"Test Loss: {avg_val_loss:.6f}")


    # Train the model
    train_losses, val_losses = train_cnn_model(model, train_loader, val_loader, criterion, optimizer, device, n_epochs)

    # Save the trained model
    model_path = "models/lane_detection_cnn7.pth"
    os.makedirs(os.path.dirname(model_path), exist_ok=True)
    torch.save(model.state_dict(), model_path)
    print(f"Model saved to {model_path}")

    # Test the model
    test_loss = test_cnn_model(model, test_loader, criterion, device)
    print(f"Test Loss: {test_loss:.6f}")
