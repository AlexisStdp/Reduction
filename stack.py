# implementations of the Stack and FFN classes
# these will hopefully then be integrated into more advanced architectures (Transformers, etc.)

import torch
import torch.nn as nn
import torch.nn.functional as F

class Stack(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim, dropout=0.1, activation=F.relu):
        super(Stack, self).__init__()
        # Define the first affine transformation + bias (V*X + B)
        self.affine1 = nn.Linear(input_dim, hidden_dim)
        # Define the second affine transformation (W*ReLU(.) + bias C implicitly included)
        self.affine2 = nn.Linear(hidden_dim, output_dim)
        # Define the skip-like learnable affine transformation (W_affine*X)
        self.affine_skip = nn.Linear(input_dim, output_dim, bias=False)
        if dropout > 0:
            self.dropout = nn.Dropout(dropout)
        self.activation = activation

    def forward(self, x):
        # First affine transformation followed by activation function
        out = self.activation(self.affine1(x))
        # Second affine transformation (bias C is included here implicitly)
        out = self.affine2(out)
        # Skip-like learnable affine transformation
        out_skip = self.affine_skip(x)
        # Final output computation (addition includes bias C implicitly from affine2)
        y = out + out_skip
        # Final formula: y = W_affine*X + W*ReLU(V*X + B) + C
        if hasattr(self, 'dropout'):
            y = self.dropout(y)
        return y
    
    def get_weights(self):
        """
        Return the weights of the model

        Format: [V, B, W, C, W_affine]

        In some notations W_affine = Q @ P 
        """
        return [self.affine1.weight.clone(), self.affine1.bias.clone(), self.affine2.weight.clone(), self.affine2.bias.clone(), self.affine_skip.weight.clone()]
    
    def set_weights(self, weights):
        """
        Set the weights of the model

        Format: [V, B, W, C, W_affine]

        In some notations W_affine = Q @ P 
        """
        # self.affine1.weight = weights[0]
        self.affine1.weight = nn.Parameter(weights[0])
        self.affine1.bias = nn.Parameter(weights[1])
        self.affine2.weight = nn.Parameter(weights[2])
        self.affine2.bias = nn.Parameter(weights[3])
        self.affine_skip.weight = nn.Parameter(weights[4])

class svdStack(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim, hidden_affine_dim=0):
        super(svdStack, self).__init__()
        # Define the first affine transformation + bias (V*X + B)
        self.affine1 = nn.Linear(input_dim, hidden_dim)
        # Define the second affine transformation (W*ReLU(.) + bias C implicitly included)
        self.affine2 = nn.Linear(hidden_dim, output_dim)
        # Define the skip-like learnable affine transformation (W_affine*X)
        self.affine_skip = nn.Linear(input_dim, hidden_affine_dim, bias=False)
        self.affine_skip2 = nn.Linear(hidden_affine_dim, output_dim, bias=False)

    def forward(self, x):
        # First affine transformation followed by ReLU
        out = F.relu(self.affine1(x))
        # Second affine transformation (bias C is included here implicitly)
        out = self.affine2(out)
        # Skip-like learnable affine transformation
        out_skip = self.affine_skip(x)
        out_skip = self.affine_skip2(out_skip)
        # Final output computation (addition includes bias C implicitly from affine2)
        y = out + out_skip
        # # # # Final formula: y = W_affine*X + W*ReLU(V*X + B) + C
        # Final formula: y = W_1*W_2*X + W*ReLU(V*X + B) + C
        return y
    
    def get_weights(self):
        """
        Return the weights of the model

        Format: [V, B, W, C, W_affine1, W_affine2]
        """
        return [self.affine1.weight.clone(), self.affine1.bias.clone(), self.affine2.weight.clone(), self.affine2.bias.clone(), self.affine_skip.weight.clone(), self.affine_skip2.weight.clone()]
    
    def set_weights(self, weights):
        """
        Set the weights of the model

        Format: [V, B, W, C, W_affine1, W_affine2]
        """
        # self.affine1.weight = weights[0]
        self.affine1.weight = nn.Parameter(weights[0])
        self.affine1.bias = nn.Parameter(weights[1])
        self.affine2.weight = nn.Parameter(weights[2])
        self.affine2.bias = nn.Parameter(weights[3])
        self.affine_skip.weight = nn.Parameter(weights[4])
        self.affine_skip2.weight = nn.Parameter(weights[5])

class FFN(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim, dropout=0.1):
        super(FFN, self).__init__()
        # Define the first affine transformation + bias (V*X + B)
        self.affine1 = nn.Linear(input_dim, hidden_dim)
        # Define the second affine transformation (W*ReLU(.) + bias C implicitly included)
        self.affine2 = nn.Linear(hidden_dim, output_dim)
        # Dropout layer
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        # First affine transformation followed by ReLU
        out = F.relu(self.affine1(x))
        # Second affine transformation (bias C is included here implicitly)
        out = self.affine2(out)
        # Residual connection
        y = out # + x # TODO: fix residual connection
        # Final formula: y = W*ReLU(V*X + B) + X 
        y = self.dropout(y)
        return y
    
    def get_weights(self):
        """
        Return the weights of the model

        Format: [V, B, W, C]
        """
        return [self.affine1.weight.clone(), self.affine1.bias.clone(), self.affine2.weight.clone(), self.affine2.bias.clone()]
    
    def set_weights(self, weights):
        """
        Set the weights of the model

        Format: [V, B, W, C]
        """
        # self.affine1.weight = weights[0]
        self.affine1.weight = nn.Parameter(weights[0])
        self.affine1.bias = nn.Parameter(weights[1])
        self.affine2.weight = nn.Parameter(weights[2])
        self.affine2.bias = nn.Parameter(weights[3])


if __name__ == '__main__':
    input_dim = 10
    hidden_dim = 5
    output_dim = 3
    model = Stack(input_dim, hidden_dim, output_dim)
    x = torch.randn(1, input_dim)
    y = model(x)
    print(f"Input shape: {x.shape}; Output shape: {y.shape}")
    print(y)