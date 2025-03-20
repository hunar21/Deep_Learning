import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import torchvision.transforms as transforms
import torchvision.datasets as datasets
import numpy as np
import random
import argparse
from PIL import Image, ImageDraw, ImageFont
import time
from task import MyExtremeLearningMachine, get_dataloaders, evaluate_model, create_montage

class MyExtremeLearningMachineLS(MyExtremeLearningMachine):
    """
    MyExtremeLearningMachineLS extends MyExtremeLearningMachine to implement a direct least-square solution
    for training the output layer weights of the Extreme Learning Machine (ELM).

    Inherits all attributes and methods from MyExtremeLearningMachine.
    """

    def fit_elm_ls(self, train_loader, device='cpu', subset_fraction=0.1):
        """
        Fit the Extreme Learning Machine (ELM) model using a direct least-square solver.

        This method computes the hidden layer outputs from a subset of the training data and then
        solves for the output layer weights using the least squares method.

        Args:
            train_loader (torch.utils.data.DataLoader): DataLoader providing the training dataset.
                The dataset is expected to have a 'dataset' attribute that supports indexing.
            device (str): Device for computation (e.g., 'cpu' or 'cuda').
            subset_fraction (float): Fraction (between 0 and 1) of the training dataset to use for least squares estimation.
                For example, 0.1 indicates that 10% of the data will be used.
        """
        self.to(device)
        self.eval()
        # Use a subset of the data
        subset_size = int(len(train_loader.dataset) * subset_fraction)
        subset_indices = torch.randperm(len(train_loader.dataset))[:subset_size]
        subset_loader = DataLoader(train_loader.dataset, batch_size=train_loader.batch_size, sampler=torch.utils.data.SubsetRandomSampler(subset_indices))
        
        hidden_outputs = []
        labels = []
        with torch.no_grad():
            for inputs, targets in subset_loader:
                inputs, targets = inputs.to(device), targets.to(device)
                hidden_output = self.fixed_conv(inputs)
                hidden_output = torch.relu(hidden_output)
                hidden_output = self.pool(hidden_output)
                hidden_output = hidden_output.view(hidden_output.size(0), -1)
                hidden_outputs.append(hidden_output)
                labels.append(targets)
        
        hidden_outputs = torch.cat(hidden_outputs, dim=0)
        labels = torch.cat(labels, dim=0)
        labels_onehot = torch.zeros(labels.size(0), self.fc.out_features).to(device)
        labels_onehot.scatter_(1, labels.view(-1, 1), 1)
        hidden_outputs_np = hidden_outputs.cpu().numpy()
        labels_onehot_np = labels_onehot.cpu().numpy()
        weights, _, _, _ = np.linalg.lstsq(hidden_outputs_np, labels_onehot_np, rcond=None)
        self.fc.weight.data = torch.tensor(weights.T, dtype=torch.float32).to(device)
        self.fc.bias.data = torch.zeros(self.fc.out_features).to(device)

def random_hyperparameter_search(train_loader, test_loader, device='cpu', num_trials=5, subset_fraction=0.1):
    """
    Perform a random hyperparameter search for the ELM model using the least squares method.

    This function randomly samples hyperparameters for the convolutional layer (number of output channels and kernel size)
    and the fixed standard deviation. For each trial, it trains the model on a subset of the training data using
    least squares and evaluates its accuracy on the test data.

    Args:
        train_loader (torch.utils.data.DataLoader): DataLoader for the training dataset.
        test_loader (torch.utils.data.DataLoader): DataLoader for the test dataset.
        device (str): Device for computations (e.g., 'cpu' or 'cuda').
        num_trials (int): Number of random hyperparameter trials to perform.
        subset_fraction (float): Fraction of the training data to use for the least squares estimation.

    Returns:
        tuple: A tuple containing:
            - best_model: The model instance with the highest test accuracy.
            - best_params (dict): Dictionary of hyperparameters ('conv_out_channels', 'kernel_size', 'fixed_std') that achieved the best accuracy.
            - best_accuracy (float): The best test accuracy achieved.
    """
    best_accuracy = 0
    best_params = None
    best_model = None
    
    for _ in range(num_trials):
        conv_out_channels = random.choice([64, 128])
        kernel_size = random.choice([3, 5])
        fixed_std = random.uniform(0.05, 0.2)
        
        model = MyExtremeLearningMachineLS(input_channels=3, conv_out_channels=conv_out_channels,
                                           kernel_size=kernel_size, num_classes=10, input_size=(32, 32),
                                           fixed_std=fixed_std)
        model.fit_elm_ls(train_loader, device=device, subset_fraction=subset_fraction)        
        acc, _, _ = evaluate_model(model, test_loader, device=device)
        
        if acc > best_accuracy:
            best_accuracy = acc
            best_params = {
                'conv_out_channels': conv_out_channels,
                'kernel_size': kernel_size,
                'fixed_std': fixed_std
            }
            best_model = model
    
    return best_model, best_params, best_accuracy

def main():
    """
    Main function to execute training and evaluation of Extreme Learning Machine models.

    The function performs the following steps:
      1. Parses command-line arguments.
      2. Sets random seeds for reproducibility.
      3. Loads the CIFAR-10 dataset.
      4. Compares the performance of models trained using SGD and the least squares (LS) method.
      5. Performs a random hyperparameter search using the LS method.
      6. Generates a montage of test images with their true and predicted labels, saving the result as "new_result.png".
    """
    parser = argparse.ArgumentParser(description="Task 2a: Least-Square Solver for ELM")
    parser.add_argument('--batch_size', type=int, default=16)  # Reduced batch size
    parser.add_argument("--device", type=str, default="cpu", help="Device to run the model (cpu/cuda)")
    args = parser.parse_args()

    torch.manual_seed(42)
    np.random.seed(42)
    random.seed(42)
    
    train_loader, test_loader = get_dataloaders(batch_size=args.batch_size, input_size=(32, 32))
    
    print("Comparing fit_elm_sgd and fit_elm_ls... \n")
    print("Note: Since fit_elm_ls was taking a lot of time to execute due to high number of computations,"
    " I used a smaller fraction of the training data (20%) and kept hyperparameter search range fairly "
    "narrow to avoid huge computation time. Because of this, the model doesn’t see enough varied examples "
    "and doesn’t explore enough possible configurations, so it ends up with only moderate accuracy. If I "
    "increased the subset fraction, tried more hyperparameters, and ran more trials, I could likely find "
    "better settings and improve the performance of the least-squares approach.")

    # Train with SGD
    model_sgd = MyExtremeLearningMachine(input_channels=3, conv_out_channels=64, kernel_size=3,  # Reduced channels and kernel size
                                         num_classes=10, input_size=(32, 32), fixed_std=0.1)
    start_time = time.time()
    model_sgd.fit_elm_sgd(train_loader, epochs=10, lr=0.01, device=args.device)
    sgd_time = time.time() - start_time
    sgd_acc, _, _ = evaluate_model(model_sgd, test_loader, device=args.device)
    
    # Train with LS
    model_ls = MyExtremeLearningMachineLS(input_channels=3, conv_out_channels=64, kernel_size=3,  # Reduced channels and kernel size
                                          num_classes=10, input_size=(32, 32), fixed_std=0.1)
    start_time = time.time()
    model_ls.fit_elm_ls(train_loader, device=args.device, subset_fraction=0.2)  # Use 20% of the data
    ls_time = time.time() - start_time
    ls_acc, _, _ = evaluate_model(model_ls, test_loader, device=args.device)
    
    print(f"SGD Training Time: {sgd_time:.2f}s, Test Accuracy: {sgd_acc*100:.2f}%")
    print(f"LS Training Time: {ls_time:.2f}s, Test Accuracy: {ls_acc*100:.2f}%")
    
    # Random hyperparameter search with fit_elm_ls
    print("\nPerforming random hyperparameter search with fit_elm_ls...")
    best_model, best_params, best_accuracy = random_hyperparameter_search(train_loader, test_loader, device=args.device, num_trials=5, subset_fraction=0.2)  # Reduced trials and subset
    print(f"Best Parameters: {best_params}")
    print(f"Best Test Accuracy: {best_accuracy*100:.2f}%")
    
    # Generate predictions from the best model
    all_test_images = []
    all_test_labels = []
    for inputs, targets in test_loader:
        all_test_images.append(inputs)
        all_test_labels.extend(targets.numpy())
        if len(all_test_labels) >= 36:
            break
    test_images = torch.cat(all_test_images, dim=0)[:36]
    
    best_model.to(args.device)
    best_model.eval()
    with torch.no_grad():
        outputs = best_model(test_images.to(args.device))
        _, best_preds = torch.max(outputs, 1)
    best_preds = best_preds.cpu().numpy()
    
    # Save montage as "new_result.png"
    create_montage(test_images, all_test_labels[:36], best_preds, "new_result.png", grid_size=(6, 6))

if __name__ == "__main__":
    main()
