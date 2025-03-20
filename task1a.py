import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np

from task import get_combinations, prepare_datasets, MyCrossEntropy, compute_accuracy, fit_logistic_sgd


class logistic_fun_learnable_M(nn.Module):
    """
    Logistic regression model with learnable polynomial degree (M)
    This model expands input features into polynomial features using combinations (up to M_max)
    and applies a gating function to each non-constant feature based on a learnable parameter M_learned.

    Attributes:
        D (int): Input feature dimension.
        M_max (int): Maximum polynomial degree allowed.
        combs (list): List of polynomial feature combinations (generated by get_combinations).
        gate_k (float): Steepness parameter for the sigmoid gating function.
        gate_offset (float): Offset parameter for the gating function.
        w (torch.nn.Parameter): Weight parameter tensor for each polynomial feature, shape (p,),
                                where p = len(combs).
        M_learned (torch.nn.Parameter): Learnable parameter controlling the effective polynomial degree.
    """
    def __init__(self, D, M_max, combs, gate_k=10.0, gate_offset=0.5):
        """
        Initialize the logistic_fun_learnable_M model.

        Args:
            D (int): Dimensionality of the input data.
            M_max (int): Maximum allowed polynomial degree.
            combs (list): List of polynomial feature combinations (from get_combinations).
            gate_k (float, optional): Steepness of the gating function. Default is 10.0.
            gate_offset (float, optional): Offset for the gating function. Default is 0.5.
        """
        super(logistic_fun_learnable_M, self).__init__()
        self.D = D
        self.M_max = M_max
        self.combs = combs
        self.gate_k = gate_k
        self.gate_offset = gate_offset
        p = len(combs)
        self.w = nn.Parameter(torch.zeros(p))
        self.M_learned = nn.Parameter(torch.tensor(2.0))
        
    def forward(self, X):
        """
        Perform the forward pass of the model.

        Args:
            X (torch.Tensor): Input tensor of shape (N, D), where N is the number of samples and D is the feature dimension.

        Returns:
            torch.Tensor: Output predictions as a tensor of shape (N,), with values in (0, 1) after sigmoid activation.
        """
        X_poly = self.poly_features_with_gate(X)
        logits = X_poly @ self.w
        y = torch.sigmoid(logits)
        return y

    def poly_features_with_gate(self, X):
        """
        Compute the gated polynomial features for the input.

        For each sample, compute its polynomial features based on self.combs.
        For each non-constant feature (degree > 0), apply a gating factor:
            gate = sigmoid(gate_k * (rounded_M - d + gate_offset))
        where d is the degree of the feature and rounded_M is the rounded value of M_learned.

        Args:
            X (torch.Tensor): Input tensor of shape (N, D), where N is the number of samples.

        Returns:
            torch.Tensor: Tensor of gated polynomial features of shape (N, p),
                          where p = len(self.combs).
        """
        features_list = []
        # Use straight-through estimator to obtain a rounded value of M_learned while preserving gradients.
        m_int = (torch.round(self.M_learned) - self.M_learned).detach() + self.M_learned
        for x in X:
            feats = []
            for comb in self.combs:
                d = len(comb)
                if d == 0:
                    feat = torch.tensor(1.0, device=x.device)
                else:
                    prod = torch.prod(x[list(comb)])
                    gate = torch.sigmoid(self.gate_k * (m_int - d + self.gate_offset))
                    feat = prod * gate
                feats.append(feat)
            feats = torch.stack(feats)
            features_list.append(feats)
        return torch.stack(features_list)


def main():
    """
    Main function to generate data, train the model, and evaluate its performance.

    Generates training and testing datasets using a fixed underlying polynomial degree,
    initializes the logistic_fun_learnable_M model, trains it using SGD, and prints the
    optimized learnable M value (rounded to an integer) along with accuracy metrics.

    No arguments or return values.
    """
    torch.manual_seed(0)
    np.random.seed(0)

    D = 5
    # Generate data using a fixed underlying polynomial degree (M_data=2)
    X_train, t_train, X_test, t_test = prepare_datasets(D, 2)
    
    M_max = 3
    combs = get_combinations(D, M_max)
    
    print("\n--- Training model with learnable M ---")
    model = logistic_fun_learnable_M(D, M_max, combs)
    loss_fn = MyCrossEntropy()
    optimal_params = fit_logistic_sgd(X_train, t_train, model, loss_fn,
                                      learning_rate=0.01, batch_size=40, epochs=400)
    train_acc = compute_accuracy(model, X_train, t_train)
    test_acc = compute_accuracy(model, X_test, t_test)
    
    print("\n--- Final Results ---")
    print(f"Optimized M value: {int(round(model.M_learned.item()))}")
    print(f"Accuracy on training data: {train_acc*100:.2f}%")
    print(f"Accuracy on test data: {test_acc*100:.2f}%")


if __name__ == "__main__":
    main()
