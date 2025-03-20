import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import torchvision.transforms as transforms
import torchvision.datasets as datasets
import numpy as np
import os
import random
import argparse
from PIL import Image, ImageDraw, ImageFont


class MyExtremeLearningMachine(nn.Module):
    """
    Implements a simple Convolutional Neural Network for Extreme Learning Machine (ELM) training.
    
    Attributes:
        input_size (tuple): Input image dimensions as (H, W).
        fixed_conv (nn.Conv2d): A fixed convolutional layer with weights initialized with a Gaussian distribution.
        pool (nn.MaxPool2d): A max pooling layer with kernel size 2x2.
        fc (nn.Linear): A fully connected layer for classification.
    
    Methods:
        initialise_fixed_layers(fixed_std):
            Initialize fixed convolution weights using a normal distribution.
        forward(x):
            Perform a forward pass through the network.
        fit_elm_sgd(train_loader, epochs, lr, device='cpu'):
            Train the final layer using SGD (backpropagation) on the given training data.
    
    Input arguments for __init__:
        input_channels (int): Number of input channels (e.g., 3 for RGB images).
        conv_out_channels (int): Number of output channels for the fixed convolution layer.
        kernel_size (int): Size of the convolution kernel (assumed square).
        num_classes (int): Number of target classes.
        input_size (tuple): Tuple (H, W) representing input image dimensions.
        fixed_std (float): Standard deviation for Gaussian initialization of fixed convolution weights.
    
    Output:
        An instance of MyExtremeLearningMachine.
    """
    def __init__(self, input_channels, conv_out_channels, kernel_size, num_classes, input_size, fixed_std):
        super(MyExtremeLearningMachine, self).__init__()
        self.input_size = input_size
        self.fixed_conv = nn.Conv2d(in_channels=input_channels, 
                                    out_channels=conv_out_channels, 
                                    kernel_size=kernel_size,
                                    padding=2, 
                                    bias=False)
        self.initialise_fixed_layers(fixed_std)
        # Freeze the fixed conv weights.
        for param in self.fixed_conv.parameters():
            param.requires_grad = False

        conv_out_height = (input_size[0] + 2*2 - kernel_size) + 1  # Account for padding
        conv_out_width = (input_size[1] + 2*2 - kernel_size) + 1
        self.pool = nn.MaxPool2d(2, 2)  # Adding MaxPool2d layer with a kernel size of 2x2
        conv_out_height //= 2  # Pooling reduces the height and width by a factor of 2
        conv_out_width //= 2

        flattened_size = conv_out_channels * conv_out_height * conv_out_width
        self.fc = nn.Linear(flattened_size, num_classes)
    
    def initialise_fixed_layers(self, fixed_std):
        """
        Initialize the weights of the fixed convolution layer.
        
        Parameters:
            fixed_std (float): Standard deviation for the normal distribution used for initialization.

        """
        nn.init.normal_(self.fixed_conv.weight, mean=0.0, std=fixed_std)
    
    def forward(self, x):
        """
        Forward pass of the network.
        
        Parameters:
            x (Tensor): Input tensor of shape (batch_size, input_channels, H, W).
        
        Returns:
            Tensor: Output logits of shape (batch_size, num_classes).
        """
        x = self.fixed_conv(x)
        x = torch.relu(x)
        x = self.pool(x)  # Apply max pooling
        x = x.view(x.size(0), -1)  # Flatten the tensor
        x = self.fc(x)
        return x

    def fit_elm_sgd(self, train_loader, epochs, lr, device='cpu'):
        """
        Train the final layer using SGD (standard backpropagation).
        
        Parameters:
            train_loader (DataLoader): DataLoader for training data.
            epochs (int): Number of training epochs.
            lr (float): Learning rate for SGD.
            device (str): Device to use for training ('cpu' or 'cuda').
        """
        self.to(device)
        optimizer = optim.SGD(self.fc.parameters(), lr=lr)
        criterion = nn.CrossEntropyLoss()
        for epoch in range(epochs):
            self.train()
            for inputs, targets in train_loader:
                inputs, targets = inputs.to(device), targets.to(device)
                optimizer.zero_grad()
                outputs = self(inputs)
                loss = criterion(outputs, targets)
                loss.backward()
                optimizer.step()


class MyMixUp:
    """
    Implements the MixUp data augmentation technique.
    
    Attributes:
        alpha (float): The alpha parameter for the Beta distribution.
        num_classes (int): Number of classes for one-hot encoding.
    
    Methods:
        mixup(x, y):
            Returns a mixup-augmented batch of inputs and targets.
    
    Input arguments for __init__:
        alpha (float): Alpha parameter for Beta distribution.
        seed (int): Seed for random number generators.
        num_classes (int): Number of target classes.
    
    Output:
        An instance of MyMixUp.
    """
    def __init__(self, alpha, seed, num_classes):
        self.alpha = alpha
        self.num_classes = num_classes
        random.seed(seed)
        np.random.seed(seed)
    
    def mixup(self, x, y):
        """
        Perform MixUp augmentation on a batch of data.
        
        Parameters:
            x (Tensor): Input tensor of shape (batch_size, ...).
            y (Tensor): Target tensor of shape (batch_size,).
        
        Returns:
            mixed_x (Tensor): Mixed input tensor of shape (batch_size, ...).
            mixed_y (Tensor): Mixed target tensor of shape (batch_size, num_classes), as a soft one-hot encoding.
        """
        batch_size = x.size(0)
        indices = torch.randperm(batch_size)
        shuffled_x = x[indices]
        shuffled_y = y[indices]
        lam = np.random.beta(self.alpha, self.alpha)
        lam = max(lam, 1 - lam)
        mixed_x = lam * x + (1 - lam) * shuffled_x
        y_onehot = torch.zeros(batch_size, self.num_classes).scatter_(1, y.view(-1,1), 1)
        shuffled_y_onehot = torch.zeros(batch_size, self.num_classes).scatter_(1, shuffled_y.view(-1,1), 1)
        mixed_y = lam * y_onehot + (1 - lam) * shuffled_y_onehot
        return mixed_x, mixed_y

class MyEnsembleELM:
    """
    Implements an ensemble of ELM models.
    
    Attributes:
        models (list): A list of MyExtremeLearningMachine instances.
    
    Methods:
        predict(x, device='cpu'):
            Returns the average prediction of all models in the ensemble.
    
    Input arguments for __init__:
        num_models (int): Number of models in the ensemble (between 1 and 20).
        input_channels (int): Number of input channels.
        conv_out_channels (int): Number of output channels for the fixed convolution.
        kernel_size (int): Kernel size for the convolution layer.
        num_classes (int): Number of target classes.
        input_size (tuple): Input image dimensions as (H, W).
        fixed_std (float): Standard deviation for Gaussian initialization of fixed conv weights.
        seed (int): Seed for reproducibility.
    
    Output:
        An instance of MyEnsembleELM.
    """
    def __init__(self, num_models, input_channels, conv_out_channels, kernel_size, 
                 num_classes, input_size, fixed_std, seed):
        if num_models < 1 or num_models > 20:
            raise ValueError("num_models should be between 1 and 20.")
        self.models = []
        torch.manual_seed(seed)
        np.random.seed(seed)
        for i in range(num_models):
            model = MyExtremeLearningMachine(input_channels, conv_out_channels, kernel_size, 
                                             num_classes, input_size, fixed_std)
            self.models.append(model)
    
    def predict(self, x, device='cpu'):
        """
        Compute the ensemble prediction by averaging softmax probabilities from all models.
        
        Parameters:
            x (Tensor): Input tensor of shape (batch_size, ...).
            device (str): Device to perform computation ('cpu' or 'cuda').
        
        Returns:
            Tensor: Averaged probability tensor of shape (batch_size, num_classes).
        """
        preds = []
        for model in self.models:
            model.to(device)
            model.eval()
            with torch.no_grad():
                outputs = model(x.to(device))
                probs = torch.softmax(outputs, dim=1)
                preds.append(probs)
        avg_probs = torch.stack(preds).mean(dim=0)
        return avg_probs


def save_mixup_examples(mixup_augmenter, train_loader, device, filename="mixup.png"):
    """
    Generates and saves a montage of 16 images after applying MixUp augmentation.
    
    Parameters:
        mixup_augmenter (MyMixUp): An instance of MyMixUp for data augmentation.
        train_loader (DataLoader): DataLoader for training data.
        device (str): Device to run the model ("cpu" or "cuda").
        filename (str): Name of the output PNG file (default: "mixup.png").
    
    Returns:
        None. Saves a montage of MixUp-augmented images to `filename`.
    """
    data_iter = iter(train_loader)
    images, labels = next(data_iter)
    images, labels = images.to(device), labels.to(device)
    
    mixed_images, mixed_labels = mixup_augmenter.mixup(images, labels)    
    top_labels = torch.argmax(mixed_labels, dim=1).cpu().numpy()    
    n_images = min(16, mixed_images.size(0))    
    create_montage(
        images=mixed_images[:n_images].cpu(), 
        labels=top_labels[:n_images], 
        predictions=top_labels[:n_images], 
        filename=filename, 
        grid_size=(4,4)  # 4 rows x 4 columns
    )
    print(f"MixUp examples saved to {filename}")

def compute_accuracy(outputs, targets):
    """
    Computes the accuracy given model outputs and true targets.
    
    Parameters:
        outputs (Tensor): Logits output by the model (shape: (batch_size, num_classes)).
        targets (Tensor): Ground truth labels (shape: (batch_size,)).
    
    Returns:
        float: Accuracy (value between 0 and 1).
    """
    _, preds = torch.max(outputs, 1)
    correct = torch.sum(preds == targets).item()
    return correct / targets.size(0)

def compute_f1_score(true, pred, num_classes):
    """
    Computes the macro F1 score for the given true and predicted labels.
    
    Parameters:
        true (array-like): Array of true labels.
        pred (array-like): Array of predicted labels.
        num_classes (int): Total number of classes.
    
    Returns:
        float: Macro F1 score.
    """
    true = np.array(true)
    pred = np.array(pred)
    f1_scores = []
    for cls in range(num_classes):
        tp = np.sum((true == cls) & (pred == cls))
        fp = np.sum((true != cls) & (pred == cls))
        fn = np.sum((true == cls) & (pred != cls))
        if tp == 0:
            f1_scores.append(0)
        else:
            precision = tp / (tp + fp) if (tp + fp) > 0 else 0
            recall = tp / (tp + fn) if (tp + fn) > 0 else 0
            if precision + recall == 0:
                f1 = 0
            else:
                f1 = 2 * precision * recall / (precision + recall)
            f1_scores.append(f1)
    return np.mean(f1_scores)

def get_dataloaders(batch_size, input_size):
    """
    Loads CIFAR-10 training and test datasets and returns corresponding DataLoaders.
    
    Parameters:
        batch_size (int): Batch size for the DataLoaders.
        input_size (tuple): Tuple (H, W) for resizing images.
    
    Returns:
        train_loader (DataLoader): DataLoader for the CIFAR-10 training dataset.
        test_loader (DataLoader): DataLoader for the CIFAR-10 test dataset.
    """
    transform = transforms.Compose([
        transforms.Resize(input_size),
        transforms.ToTensor(),
        transforms.Normalize((0.5,0.5,0.5), (0.5,0.5,0.5))
    ])
    train_dataset = datasets.CIFAR10(root='./data', train=True, download=True, transform=transform)
    test_dataset = datasets.CIFAR10(root='./data', train=False, download=True, transform=transform)
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)
    return train_loader, test_loader

def evaluate_model(model, test_loader, device='cpu'):
    """
    Evaluates a given model on the test dataset.
    
    Parameters:
        model (nn.Module): The model to evaluate.
        test_loader (DataLoader): DataLoader for the test dataset.
        device (str): Device to perform evaluation on ('cpu' or 'cuda').
    
    Returns:
        accuracy (float): Test accuracy (value between 0 and 1).
        all_preds (list): List of predicted labels for the test set.
        all_labels (list): List of ground truth labels for the test set.
    """
    model.to(device)
    model.eval()
    total_acc = 0.0
    count = 0
    all_preds = []
    all_labels = []
    with torch.no_grad():
        for inputs, targets in test_loader:
            inputs, targets = inputs.to(device), targets.to(device)
            outputs = model(inputs)
            acc = compute_accuracy(outputs, targets)
            total_acc += acc * inputs.size(0)
            count += inputs.size(0)
            _, preds = torch.max(outputs, 1)
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(targets.cpu().numpy())
    return total_acc / count, all_preds, all_labels

def evaluate_ensemble(ensemble, test_loader, device='cpu'):
    """
    Evaluates an ensemble of models on the test dataset.
    
    Parameters:
        ensemble (MyEnsembleELM): An ensemble of ELM models.
        test_loader (DataLoader): DataLoader for the test dataset.
        device (str): Device for evaluation ('cpu' or 'cuda').
    
    Returns:
        accuracy (float): Test accuracy of the ensemble.
        all_preds (list): List of predicted labels.
        all_labels (list): List of ground truth labels.
    """
    total_acc = 0.0
    count = 0
    all_preds = []
    all_labels = []
    with torch.no_grad():
        for inputs, targets in test_loader:
            inputs, targets = inputs.to(device), targets.to(device)
            avg_probs = ensemble.predict(inputs, device=device)
            _, preds = torch.max(avg_probs, 1)
            total_acc += torch.sum(preds == targets).item()
            count += inputs.size(0)
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(targets.cpu().numpy())
    return total_acc / count, all_preds, all_labels

def train_model_variant(model, variant_name, train_loader, test_loader, epochs, lr, device, mixup_augmenter=None):
    """
    Trains a single ELM model variant using SGD (and optionally MixUp) and evaluates it after each epoch.
    
    Parameters:
        model (nn.Module): The ELM model to train.
        variant_name (str): Name of the model variant (used for printing/logging).
        train_loader (DataLoader): DataLoader for training data.
        test_loader (DataLoader): DataLoader for testing data.
        epochs (int): Number of epochs to train.
        lr (float): Learning rate for SGD.
        device (str): Device to use for training ('cpu' or 'cuda').
        mixup_augmenter (MyMixUp, optional): An instance of MyMixUp for data augmentation (default: None).
    
    Returns:
        history (list): A list of dictionaries containing epoch, loss, accuracy, and F1 score.
        model (nn.Module): The trained model.
    """
    history = []
    optimizer = optim.SGD(model.fc.parameters(), lr=lr)
    for epoch in range(epochs):
        model.train()
        running_loss = 0.0
        for inputs, targets in train_loader:
            inputs, targets = inputs.to(device), targets.to(device)
            optimizer.zero_grad()
            if mixup_augmenter:
                mixed_inputs, mixed_targets = mixup_augmenter.mixup(inputs, targets)
                outputs = model(mixed_inputs)
                loss = nn.KLDivLoss()(torch.log_softmax(outputs, dim=1), mixed_targets.to(device))
            else:
                outputs = model(inputs)
                loss = nn.CrossEntropyLoss()(outputs, targets)
            loss.backward()
            optimizer.step()
            running_loss += loss.item()
        avg_loss = running_loss / len(train_loader)
        acc, preds, true_labels = evaluate_model(model, test_loader, device)
        f1 = compute_f1_score(true_labels, preds, len(train_loader.dataset.classes))
        print(f"{variant_name} Epoch {epoch+1}/{epochs}: Loss={avg_loss:.4f}, Test Acc={acc*100:.2f}%, F1={f1:.4f}")
        history.append({"epoch": epoch+1, "loss": avg_loss, "accuracy": acc, "f1": f1})
    return history, model

def train_ensemble_variant(ensemble, variant_name, train_loader, test_loader, epochs, lr, device, mixup_augmenter=None):
    """
    Trains an ensemble of ELM models, each model being trained individually for one epoch per round,
    and evaluates the ensemble after each epoch.
    
    Parameters:
        ensemble (MyEnsembleELM): The ensemble of ELM models.
        variant_name (str): Name of the ensemble variant (used for printing/logging).
        train_loader (DataLoader): DataLoader for training data.
        test_loader (DataLoader): DataLoader for testing data.
        epochs (int): Number of training epochs.
        lr (float): Learning rate for SGD.
        device (str): Device to use ('cpu' or 'cuda').
        mixup_augmenter (MyMixUp, optional): An instance of MyMixUp for data augmentation (default: None).
    
    Returns:
        history (list): A list of dictionaries containing epoch, loss, accuracy, and F1 score.
        ensemble (MyEnsembleELM): The trained ensemble of models.
    """
    history = []
    # Create an optimizer for each model
    optimizers = [optim.SGD(elm.fc.parameters(), lr=lr) for elm in ensemble.models]
    for epoch in range(epochs):
        # Train each model for one epoch
        for i, elm in enumerate(ensemble.models):
            elm.train()
            running_loss = 0.0
            for inputs, targets in train_loader:
                inputs, targets = inputs.to(device), targets.to(device)
                optimizers[i].zero_grad()
                if mixup_augmenter:
                    mixed_inputs, mixed_targets = mixup_augmenter.mixup(inputs, targets)
                    outputs = elm(mixed_inputs)
                    loss = nn.KLDivLoss()(torch.log_softmax(outputs, dim=1), mixed_targets.to(device))
                else:
                    outputs = elm(inputs)
                    loss = nn.CrossEntropyLoss()(outputs, targets)
                loss.backward()
                optimizers[i].step()
                running_loss += loss.item()
        # Evaluate ensemble after each epoch
        acc, preds, true_labels = evaluate_ensemble(ensemble, test_loader, device)
        f1 = compute_f1_score(true_labels, preds, len(train_loader.dataset.classes))
        avg_loss = running_loss / len(train_loader)  # approximate loss from last model
        print(f"{variant_name} Ensemble Epoch {epoch+1}/{epochs}: Loss~{avg_loss:.4f}, Test Acc={acc*100:.2f}%, F1={f1:.4f}")
        history.append({"epoch": epoch+1, "loss": avg_loss, "accuracy": acc, "f1": f1})
    return history, ensemble

def load_or_train_model(model, model_path, variant_name, train_loader, test_loader, epochs, lr, device, mixup_augmenter=None):
    """
    Loads a pre-trained model from disk if available; otherwise, trains the model using train_model_variant
    and saves it to disk.
    
    Parameters:
        model (nn.Module): The model to load or train.
        model_path (str): File path to load/save the model state.
        variant_name (str): A string name indicating the model variant.
        train_loader (DataLoader): DataLoader for training data.
        test_loader (DataLoader): DataLoader for testing data.
        epochs (int): Number of training epochs.
        lr (float): Learning rate for SGD.
        device (str): Device for training ('cpu' or 'cuda').
        mixup_augmenter (MyMixUp, optional): Instance of MyMixUp for data augmentation (default: None).
    
    Returns:
        model (nn.Module): The loaded or newly trained model.
    """
    if os.path.exists(model_path):
        print(f"Loading pre-trained {variant_name} model from {model_path}...")
        model.load_state_dict(torch.load(model_path, map_location=device))
        return model
    else:
        print(f"No pre-trained {variant_name} model found. Training a new one...")
        history, trained_model = train_model_variant(model, variant_name, train_loader, test_loader, epochs, lr, device, mixup_augmenter)
        torch.save(trained_model.state_dict(), model_path)
        print(f"Saved {variant_name} model to {model_path}")
        return trained_model

def load_or_train_ensemble(ensemble, model_paths, variant_name, train_loader, test_loader, epochs, lr, device, mixup_augmenter=None):
    """
    Loads pre-trained ensemble models from disk if available; otherwise, trains the ensemble using train_ensemble_variant
    and saves each model's state to disk.
    
    Parameters:
        ensemble (MyEnsembleELM): The ensemble of ELM models.
        model_paths (list): A list of file paths for each model's state.
        variant_name (str): Name of the ensemble variant.
        train_loader (DataLoader): DataLoader for training data.
        test_loader (DataLoader): DataLoader for testing data.
        epochs (int): Number of training epochs.
        lr (float): Learning rate for SGD.
        device (str): Device for training ('cpu' or 'cuda').
        mixup_augmenter (MyMixUp, optional): Instance of MyMixUp for data augmentation (default: None).
    
    Returns:
        ensemble (MyEnsembleELM): The loaded or newly trained ensemble.
    """
    all_models_exist = all(os.path.exists(path) for path in model_paths)
    
    if all_models_exist:
        print(f"Loading pre-trained {variant_name} ensemble models...")
        for idx, path in enumerate(model_paths):
            ensemble.models[idx].load_state_dict(torch.load(path, map_location=device))
        return ensemble
    else:
        print(f"No pre-trained {variant_name} ensemble found. Training new models...")
        history, trained_ensemble = train_ensemble_variant(ensemble, variant_name, train_loader, test_loader, epochs, lr, device, mixup_augmenter)
        for idx, model in enumerate(trained_ensemble.models):
            torch.save(model.state_dict(), model_paths[idx])
            print(f"Saved {variant_name} ensemble model {idx} to {model_paths[idx]}")
        return trained_ensemble

def create_montage(images, labels, predictions, filename, grid_size=(6,6)):
    """
    Creates and saves a montage image displaying input images with their true (T) and predicted (P) labels
    at the bottom of each tile.

    Parameters:
        images (Tensor): A tensor containing image data (shape: (N, C, H, W)).
        labels (list or array): List of true labels for the images.
        predictions (list or array): List of predicted labels for the images.
        filename (str): The filename to save the montage image.
        grid_size (tuple): Tuple (rows, columns) specifying the grid dimensions.

    Returns:
        None. Saves the montage image to the specified filename.
    """
    import torchvision.transforms as transforms
    from PIL import Image, ImageDraw, ImageFont
    
    to_pil = transforms.ToPILImage()
    sample_img = to_pil(images[0].cpu())
    img_width, img_height = sample_img.size
    grid_rows, grid_cols = grid_size  # e.g. (6, 6)
    
    montage_width = grid_cols * img_width
    montage_height = grid_rows * img_height
    montage_image = Image.new(mode="RGB", size=(montage_width, montage_height), color=(255, 255, 255))
    
    try:
        font = ImageFont.truetype("arial.ttf", 12)
    except IOError:
        font = ImageFont.load_default()
        
    draw = ImageDraw.Draw(montage_image)
    
    for idx in range(grid_rows * grid_cols):
        if idx >= len(images):
            break
        
        # Convert the idx-th image to PIL
        img = to_pil(images[idx].cpu())
        
        # Determine row and column in the grid
        row = idx // grid_cols
        col = idx % grid_cols
        
        # Calculate offsets for placing the image
        x_offset = col * img_width
        y_offset = row * img_height
        
        # Paste the image tile onto the montage
        montage_image.paste(img, (x_offset, y_offset))
        
        # Prepare the text overlay
        text = f"T:{labels[idx]} P:{predictions[idx]}"
        
        # Place the overlay at the bottom of the tile (rather than the top)
        overlay_y = y_offset + (img_height - 15)  # 15px high text box
        overlay = Image.new('RGB', (img_width, 15), (255, 255, 255))
        montage_image.paste(overlay, (x_offset, overlay_y))
        
        # Draw the text
        draw.text((x_offset + 2, overlay_y), text, fill=(0, 0, 0), font=font)
    
    montage_image.save(filename)
    print(f"Saved montage to {filename}")


def main():
    """
    Main function to execute the training, evaluation, and selection of the best model variant.
    
    The function:
        - Parses command-line arguments.
        - Sets random seeds for reproducibility.
        - Loads the CIFAR-10 dataset.
        - Provides explanations for baseline random guess and chosen evaluation metrics.
        - Loads or trains four variants of ELM models:
            1. Baseline (no regularisation)
            2. With MixUp regularisation
            3. Ensemble without MixUp
            4. Ensemble with MixUp
        - Evaluates each variant, selects the best based on test accuracy, and generates a montage of predictions.
    
    Input arguments (via argparse):
        --epochs (int): Number of epochs for training.
        --lr (float): Learning rate for training.
        --batch_size (int): Batch size for the DataLoaders.
        --device (str): Device to run the model ("cpu").
    """
    parser = argparse.ArgumentParser(description="Task 2: Regularising Extreme Learning Machines")
    parser.add_argument('--epochs', type=int, default=10)
    parser.add_argument('--lr', type=float, default=0.01)
    parser.add_argument('--batch_size', type=int, default=64)
    parser.add_argument("--device", type=str, default="cpu", help="Device to run the model (cpu/cuda)")
    args = parser.parse_args()

    # Set seeds for reproducibility.
    torch.manual_seed(42)
    np.random.seed(42)
    random.seed(42)
    
    # Load CIFAR-10 dataset.
    train_loader, test_loader = get_dataloaders(batch_size=args.batch_size, input_size=(32,32))
    num_classes = len(train_loader.dataset.classes)
    input_channels = 3
    input_size = (32,32)
    conv_out_channels = 64
    kernel_size = 3
    fixed_std = .1

    # --- Explanation of Random Guess ---
    print("\nRandom Guess Explanation:")
    print("In a 10-class problem (like CIFAR-10), a random guess would assign each class "
    "a 10% probability on average. We can test 'randomness' by comparing model "
    "accuracy to ~10%. Any performance significantly above 10% indicates the model "
    "has learned to identify meaningful features beyond random guessing.")

    # --- Explanation of Chosen Metrics ---
    print("\nChosen Metrics: Accuracy and F1 Score.")
    print("Accuracy measures the fraction of correct predictions across all samples. "
    "However, if certain classes are under-represented, Accuracy alone can be "
    "misleading. Macro F1 treats each class equally, computing precision and recall "
    "per class and then averaging. This helps gauge performance on minority classes. \n" )
    
    print("Note: Saved models are present in saved_models. If you wanna retrain the models, delete the directory saved_models, tune your "
    "parameters and run the script again")
    # Directory to store trained models
    model_dir = "saved_models"
    os.makedirs(model_dir, exist_ok=True)

    # File paths for models
    baseline_model_path = os.path.join(model_dir, "model_baseline.pth")
    mixup_model_path = os.path.join(model_dir, "model_mixup.pth")
    ensemble_paths = [os.path.join(model_dir, f"model_ensemble_{i}.pth") for i in range(5)]
    ensemble_mixup_paths = [os.path.join(model_dir, f"model_ensemble_mixup_{i}.pth") for i in range(5)]

    # 1. Baseline ELM (no regularisation)
    print("\nChecking Baseline ELM Model...")
    baseline_model = MyExtremeLearningMachine(input_channels, conv_out_channels, kernel_size, num_classes, input_size, fixed_std)
    baseline_model = load_or_train_model(baseline_model, baseline_model_path, "Baseline", train_loader, test_loader, args.epochs, args.lr, args.device)
    print()

    # 2. ELM with MixUp Regularisation only.
    print("\nChecking ELM with MixUp...")
    mixup_augmenter = MyMixUp(alpha=0.2, seed=42, num_classes=num_classes)
    save_mixup_examples(mixup_augmenter, train_loader, args.device, filename="mixup.png")
    mixup_model = MyExtremeLearningMachine(input_channels, conv_out_channels, kernel_size, num_classes, input_size, fixed_std)
    mixup_model = load_or_train_model(mixup_model, mixup_model_path, "MixUp", train_loader, test_loader, args.epochs, args.lr, args.device, mixup_augmenter)
    print()

    # 3. Ensemble ELM (without MixUp)
    print("\nChecking Ensemble ELM (No MixUp)...")
    ensemble_model = MyEnsembleELM(num_models=5, input_channels=input_channels, conv_out_channels=conv_out_channels,
                                   kernel_size=kernel_size, num_classes=num_classes, input_size=input_size, fixed_std=fixed_std, seed=42)
    ensemble_model = load_or_train_ensemble(ensemble_model, ensemble_paths, "Ensemble", train_loader, test_loader, args.epochs, args.lr, args.device)
    print()

    # 4. Ensemble ELM with MixUp
    print("\nChecking Ensemble ELM with MixUp...")
    ensemble_mixup_model = MyEnsembleELM(num_models=5, input_channels=input_channels, conv_out_channels=conv_out_channels,
                                         kernel_size=kernel_size, num_classes=num_classes, input_size=input_size, fixed_std=fixed_std, seed=42)
    ensemble_mixup_model = load_or_train_ensemble(ensemble_mixup_model, ensemble_mixup_paths, "Ensemble+MixUp", train_loader, test_loader, args.epochs, args.lr, args.device, mixup_augmenter)
    print()

    # --- Select Best Model Variant ---
    print("\nSelecting Best Model Based on Test Accuracy...")
    variants = {
        "Baseline": baseline_model,
        "MixUp": mixup_model,
        "Ensemble": ensemble_model,
        "Ensemble+MixUp": ensemble_mixup_model
    }
    
    best_model_name = None
    best_accuracy = 0
    best_model = None

    for name, model in variants.items():
        print(f"Evaluating {name} model...")
        acc, preds, true_labels = evaluate_model(model, test_loader, args.device) if name != "Ensemble" and name != "Ensemble+MixUp" else evaluate_ensemble(model, test_loader, args.device)
        print(f"{name} Model Accuracy: {acc*100:.2f}%")
        print()
        
        if acc > best_accuracy:
            best_accuracy = acc
            best_model_name = name
            best_model = model

    print(f"\nBest model variant based on test accuracy: {best_model_name}")

    # Generate predictions from the best model
    all_test_images = []
    all_test_labels = []
    for inputs, targets in test_loader:
        all_test_images.append(inputs)
        all_test_labels.extend(targets.numpy())
        if len(all_test_labels) >= 36:
            break
    test_images = torch.cat(all_test_images, dim=0)[:36]

    if best_model_name == "Baseline" or best_model_name == "MixUp":
        best_model.to(args.device)
        best_model.eval()
        with torch.no_grad():
            outputs = best_model(test_images.to(args.device))
            _, best_preds = torch.max(outputs, 1)
        best_preds = best_preds.cpu().numpy()
    else:  # "Ensemble" or "Ensemble+MixUp"
        _, best_preds, _ = evaluate_ensemble(best_model, DataLoader(datasets.CIFAR10(root='./data', train=False, transform=transforms.Compose([
                        transforms.Resize(input_size),
                        transforms.ToTensor(),
                        transforms.Normalize((0.5,0.5,0.5), (0.5,0.5,0.5))
                    ])), batch_size=36, shuffle=False), device=args.device)

    # Save montage as "result.png"
    create_montage(test_images, all_test_labels[:36], best_preds, "result.png", grid_size=(6,6))


if __name__ == "__main__":
    main()
