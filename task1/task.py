"""
Generative GenAI Usage Statement: Generative AI tools were used in an assistive capacity for this coursework.

Specifically, AI-assisted suggestions were used for improving syntactical correctness in Python, refining the 
structure of loss function (MyCrossEntropy), and ensuring best practices in implementing stochastic gradient descent (SGD).

Additionally, AI was consulted for best practices in implementing polynomial feature expansion. I took an idea on how to do 
it and then did the implementation myself.

Finally, GenAI was used for docstrings (descriptions, inputs, outputs) for all the functions
"""

import torch
import torch.nn as nn
import torch.optim as optim
import itertools
import math
import numpy as np

def prepare_datasets(D, M):
    """
    Generates training and test datasets for logistic regression with polynomial feature expansion.
    
    Args:
        D (int): Dimensionality of input features (number of features per data point).
        M (int): Maximum polynomial order for feature expansion (positive integer).
        
    Returns:
        tuple:
            - X_train (torch.Tensor): Training input features of shape (200, D).
            - t_train (torch.Tensor): Training binary labels of shape (200,).
            - X_test (torch.Tensor): Test input features of shape (100, D).
            - t_test (torch.Tensor): Test binary labels of shape (100,).
    """
    sample_sizes = {'train': 200, 'test': 100}
    combs = get_combinations(D, M)
    p = len(combs)
    w_true = create_underlying_weights(p)
    X_train, t_train = generate_data(sample_sizes['train'], D, M, combs, w_true)
    X_test, t_test = generate_data(sample_sizes['test'], D, M, combs, w_true)
    return X_train, t_train, X_test, t_test


def get_combinations(D, M):
    """
    Generates all polynomial feature combinations (as index tuples) up to the specified order.
    
    Args:
        D (int): Dimensionality of input features (number of features).
        M (int): Maximum polynomial order (positive integer).
    
    Returns:
        list of tuple:
            Each tuple contains indices (int) representing a combination of features.
            The returned list includes an empty tuple for the bias term.
    """
    combs = [()]  # Bias term (constant feature)
    for degree in range(1, M + 1):
        combs.extend(itertools.combinations_with_replacement(range(D), degree))
    return combs


def create_underlying_weights(p):
    """
    Creates the underlying weight vector using a mathematical formula.
    
    Args:
        p (int): Total number of polynomial terms (length of weight vector).
    
    Returns:
        torch.Tensor:
            Weight vector of shape (p,), where each element is computed based on the given formula.
    """
    weights = []
    for i in range(p):
        exponent = p - i
        weight = ((-1) ** exponent) * math.sqrt(p - i) / p
        weights.append(weight)
    return torch.tensor(weights, dtype=torch.float32)


def generate_data(N, D, M, combs, w_true):
    """
    Generates synthetic data for binary logistic regression with polynomial feature expansion.
    
    Args:
        N (int): Number of data points to generate.
        D (int): Dimensionality of the input features (number of features per data point).
        M (int): Polynomial order (used in generating polynomial combinations).
        combs (list of tuple): List of feature index combinations for polynomial expansion.
        w_true (torch.Tensor): True weight vector of shape (p,), where p equals len(combs).
    
    Returns:
        tuple:
            - X (torch.Tensor): Generated input features of shape (N, D) with values uniformly sampled from [-5, 5].
            - t (torch.Tensor): Binary labels of shape (N,), generated using a noisy sigmoid transformation.
    """
    X = torch.FloatTensor(N, D).uniform_(-5.0, 5.0)
    X_poly = poly_features_batch(X, combs)
    logits = X_poly @ w_true
    y_true = torch.sigmoid(logits)
    noise = torch.randn(N)
    y_noisy = y_true + noise
    t = (y_noisy >= 0.5).float()
    return X, t


def poly_features_batch(X, combs):
    """
    Expands a batch of input vectors into polynomial features using provided combinations.
    
    Args:
        X (torch.Tensor): Input tensor of shape (N, D) where N is number of samples and D is number of features.
        combs (list of tuple): List of feature index combinations for polynomial expansion.
    
    Returns:
        torch.Tensor:
            Tensor of polynomial-expanded features of shape (N, p) where p = len(combs).
    """
    features_list = [poly_features_single(x, combs) for x in X]
    return torch.stack(features_list)


def poly_features_single(x, combs):
    """
    Expands a single input vector into its polynomial feature representation.
    
    Args:
        x (torch.Tensor): Input vector of shape (D,), where D is the dimensionality.
        combs (list of tuple): List of feature index combinations for polynomial expansion.
    
    Returns:
        torch.Tensor:
            Expanded feature vector of shape (p,), where p = len(combs). The first element corresponds to the bias term.
    """
    features = []
    for comb in combs:
        if len(comb) == 0:
            # Bias term
            features.append(torch.tensor(1.0, device=x.device))
        else:
            prod = torch.prod(x[list(comb)])
            features.append(prod)
    return torch.stack(features)


class logistic_fun(nn.Module):
    """
    Logistic regression model with polynomial feature expansion.
    
    Attributes:
        D (int): Dimensionality of input features.
        M (int): Maximum polynomial order.
        combs (list of tuple): List of feature index combinations for polynomial expansion.
        w (nn.Parameter): Model weights of shape (p,), where p = len(combs).
    """
    def __init__(self, D, M, combs):
        """
        Initializes the logistic regression model.
        
        Args:
            D (int): Dimensionality of input features.
            M (int): Maximum polynomial order.
            combs (list of tuple): List of feature index combinations for polynomial expansion.
        """
        super(logistic_fun, self).__init__()
        self.D = D
        self.M = M
        self.combs = combs
        p = len(combs)
        # Model weights are initialized as parameters.
        self.w = nn.Parameter(torch.zeros(p))
    def forward(self, X):
        """
        Performs the forward pass of the logistic regression model.
        
        Args:
            X (torch.Tensor): Input features of shape (N, D).
        
        Returns:
            torch.Tensor:
                Predicted probabilities for each sample of shape (N,), computed using the sigmoid of the logits.
        """
        X_poly = poly_features_batch(X, self.combs)
        logits = X_poly @ self.w
        y = torch.sigmoid(logits)
        return y


class MyCrossEntropy:
    def __call__(self, y, t):
        """
        Computes the mean cross-entropy loss between predicted probabilities and true binary labels.
        
        Args:
            y (torch.Tensor): Predicted probabilities of shape (N,), where N is the number of samples.
            t (torch.Tensor): True binary labels of shape (N,). Expected values are 0 or 1.
        
        Returns:
            torch.Tensor:
                A scalar tensor representing the mean cross-entropy loss.
        """
        eps = 1e-7
        y = torch.clamp(y, eps, 1 - eps)
        loss = -(t * torch.log(y) + (1 - t) * torch.log(1 - y))
        return loss.mean()


class MyRootMeanSquare:
    def __call__(self, y, t):
        """
        Computes the RMSE between predictions and targets.
        
        Args:
            y (torch.Tensor): Predicted values of shape (N,).
            t (torch.Tensor): True values of shape (N,).
        
        Returns:
            torch.Tensor:
                A scalar tensor representing the RMSE.
        """
        return torch.sqrt(torch.mean((y - t) ** 2))


def fit_logistic_sgd(X_train, t_train, model, loss_fn, learning_rate, batch_size, epochs):
    """
    Trains the logistic regression model using stochastic gradient descent (SGD) on the training dataset.
    
    Args:
        X_train (torch.Tensor): Training input features of shape (N_train, D).
        t_train (torch.Tensor): Training binary labels of shape (N_train,).
        model (logistic_fun): Instance of the logistic regression model.
        loss_fn (callable): Loss function (e.g., MyCrossEntropy or MyRootMeanSquare).
        learning_rate (float): Learning rate for SGD.
        batch_size (int): Size of mini-batches for training.
        epochs (int): Number of passes over the entire training dataset.
    
    Returns:
        torch.Tensor:
            The optimized model weights (final value of model.w) as a tensor.
    """
    optimizer = optim.SGD(model.parameters(), lr=learning_rate)
    dataset = torch.utils.data.TensorDataset(X_train, t_train)
    dataloader = torch.utils.data.DataLoader(dataset, batch_size=batch_size, shuffle=True)
    total_steps = epochs * len(dataloader)
    print_interval = max(1, total_steps // 10)
    step = 0

    for epoch in range(epochs):
        for X_batch, t_batch in dataloader:
            optimizer.zero_grad()
            y_pred = model(X_batch)
            loss = loss_fn(y_pred, t_batch)
            loss.backward()
            optimizer.step()

            if step % print_interval == 0:
                print(f"Epoch {epoch+1}, Step {step}, Loss: {loss.item():.4f}")
            step += 1

    # Final loss on training data
    final_loss = loss_fn(model(X_train), t_train)
    print(f"Final Loss on training set: {final_loss.item():.4f}")
    return model.w.data


def compute_accuracy(model, X, t):
    """
    Computes the classification accuracy of the logistic regression model.
    
    Args:
        model (logistic_fun): Instance of the logistic regression model.
        X (torch.Tensor): Input features of shape (N, D).
        t (torch.Tensor): True binary labels of shape (N,).
    
    Returns:
        float:
            The accuracy as a fraction between 0 and 1.
    """
    with torch.no_grad():
        y_pred = model(X)
        preds = (y_pred >= 0.5).float()
        accuracy = (preds == t).float().mean().item()
    return accuracy


def main():
    """
    Main function to run experiments on logistic regression with varying polynomial orders and loss functions.
    
    Generates synthetic training and test datasets, trains the model using SGD with two different loss functions 
    (Cross-Entropy and RMSE) over polynomial orders M=1,2,3, and prints out the training and test accuracies along 
    with descriptive remarks on performance.
    """
    torch.manual_seed(0)
    np.random.seed(0)

    # Generate datasets (using D = 5 and M = 2 for the underlying true data generation)
    X_train, t_train, X_test, t_test = prepare_datasets(5, 2)
    
    # Iterating over polynomial orders M = 1, 2, 3 for experiments
    for M in [1, 2, 3]:
        print(f"\n ----- Experiment with M = {M} ----- ")
        combs = get_combinations(5, M)

        for loss_name, loss_fn in [("CrossEntropy", MyCrossEntropy()), ("RMSE", MyRootMeanSquare())]:
            print(f"\nTraining with loss: {loss_name}")
            # Initialize model with weights starting at zero.
            model = logistic_fun(5, M, combs)
            # Train the model using SGD.
            optimal_w = fit_logistic_sgd(X_train, t_train, model, loss_fn,
                                         learning_rate=0.01, batch_size=40, epochs=400)
            # Evaluate accuracy on both training and test datasets.
            train_acc = compute_accuracy(model, X_train, t_train)
            test_acc = compute_accuracy(model, X_test, t_test)
            print(f"Accuracy on training data: {train_acc * 100:.2f}%")
            print(f"Accuracy on test data: {test_acc * 100:.2f}%")
    print()        
    print("Accuracy measures how well the model classifies data by comparing predicted labels with true labels. Since this is a binary classification problem, accuracy provides a simple, interpretable measure of overall model performance. However, for imbalanced datasets, additional metrics like F1-score may be more informative.")
    print()
    print("Training accuracy is usually higher than test accuracy due to overfitting, where the model memorizes training data but generalizes poorly to new data. A large difference may indicate overfitting, while similar values suggest good generalization. Cross-Entropy often yields better classification performance than RMSE since RMSE treats probabilities as continuous values rather than discrete class labels.")
    print()

if __name__ == "__main__":
    main()
