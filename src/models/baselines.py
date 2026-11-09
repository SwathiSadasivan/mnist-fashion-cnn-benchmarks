import torch
import torch.nn as nn
from sklearn.linear_model import LogisticRegression
import lightgbm as lgb


class BaselineFNN(nn.Module):
    def __init__(self, input_dim: int = 784, hidden_dim: int = 128, num_classes: int = 10):
        super(BaselineFNN, self).__init__()
        self.fc1 = nn.Linear(input_dim, hidden_dim)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(hidden_dim, num_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x.view(x.size(0), -1)
        x = self.relu(self.fc1(x))
        x = self.fc2(x)
        return x


def get_glm_baseline():
    return LogisticRegression(max_iter=100, solver="lbfgs")


def get_gbm_baseline():
    return lgb.LGBMClassifier(n_estimators=100, learning_rate=0.1, random_state=42)