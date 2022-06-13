import torch
import torch.nn as nn


""" Quadratic Neurons & Quadratic Neural Networks """


class QuadraticNeurons(nn.Module):
    """
    Define the quadratic neurons, used to construct the quadratic neurol network
    """
    def __init__(self, in_features, out_features) -> None:
        super(QuadraticNeurons, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.weight_r = nn.Parameter(torch.empty(in_features, out_features))
        self.weight_g = nn.Parameter(torch.empty(in_features, out_features))
        self.weight_b = nn.Parameter(torch.empty(in_features, out_features))
        self.bias_r = nn.Parameter(torch.zeros(out_features))
        self.bias_g = nn.Parameter(torch.zeros(out_features))
        self.bias_b = nn.Parameter(torch.zeros(out_features))
        self.reset_parameters()

    def reset_parameters(self) -> None:
        nn.init.kaiming_normal_(self.weight_r, nonlinearity='relu')
        nn.init.kaiming_normal_(self.weight_g, nonlinearity='relu')
        nn.init.kaiming_normal_(self.weight_b, nonlinearity='relu')

    def forward(self, x):  # x is the input data of current layer
        z_r = torch.mm(x, self.weight_r) + self.bias_r
        z_g = torch.mm(x, self.weight_g) + self.bias_g
        z_b = torch.mm(x ** 2, self.weight_b) + self.bias_b
        return torch.mul(z_r, z_g) + z_b

    def extra_repr(self) -> str:
        return 'in_features={}, out_features={}'.format(
            self.in_features, self.out_features
        )


class QuadraticNN1(nn.Module):
    """
    Quadratic Neural Network with
    no hidden layer (one layer of quadratic neurons)
    """
    def __init__(self, input_size, num_classes) -> None:
        super(QuadraticNN1, self).__init__()
        self.fc = QuadraticNeurons(input_size, num_classes)

    def forward(self, x):
        return self.fc(x)


class QuadraticNN2(nn.Module):
    """
    Quadratic Neural Network with
    one hidden layer (two layers of quadratic neurons)
    """
    def __init__(self, input_size, hidden_size, num_classes) -> None:
        super(QuadraticNN2, self).__init__()
        self.fc1 = QuadraticNeurons(input_size, hidden_size)
        self.relu = nn.ReLU()
        self.fc2 = QuadraticNeurons(hidden_size, num_classes)

    def forward(self, x):
        out = self.fc1(x)
        out = self.relu(out)
        out = self.fc2(out)
        return out


""" Quadratic Dendrites & Quadratic Dendritic Networks """
