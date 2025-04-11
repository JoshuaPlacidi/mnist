import torch


class MLP(torch.nn.Module):
    def __init__(
        self,
        input_dim: int,
        hidden_dim: int,
        output_dim: int,
        num_layers: int,
        activation_fn: str,
        dropout: float,
    ):
        super().__init__()
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.output_dim = output_dim
        self.num_layers = num_layers
        self.dropout = dropout

        if activation_fn == "relu":
            self.activation_fn = torch.nn.ReLU()
        elif activation_fn == "tanh":
            self.activation_fn = torch.nn.Tanh()
        elif activation_fn == "sigmoid":
            self.activation_fn = torch.nn.Sigmoid()
        else:
            raise ValueError(f"Activation function {activation_fn} not supported, must be one of: relu, tanh, sigmoid")

        self.layers = torch.nn.ModuleList()

        # input layers
        self.layers.append(torch.nn.Linear(input_dim, hidden_dim))

        # create hidden layers
        for _ in range(num_layers - 1):
            self.layers.append(torch.nn.Linear(hidden_dim, hidden_dim))

        # create output layer
        self.output_layer = torch.nn.Linear(hidden_dim, output_dim)
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:

        # flatten input
        x = x.view(x.size(0), -1)

        for layer in self.layers:
            x = self.activation_fn(layer(x))
            x = torch.nn.Dropout(self.dropout)(x)

        x = self.output_layer(x)

        return x
        
        