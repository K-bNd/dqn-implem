from torch import nn
from torch.optim import RMSprop
from hyperparameters import RMS_ESP, RMS_GRADIENT_MOMENTUM, RMS_LEARNING_RATE
import wandb

class CNN(nn.Module):
    """
    CNN model for DQN implementation
    """

    def __init__(self, game_inputs=4):
        super(CNN, self).__init__()
        
        self.cnn = nn.Sequential(
            nn.Conv2d(3, 32, kernel_size=8, stride=4),
            nn.ReLU(inplace=True),
            nn.Conv2d(32, 64, kernel_size=4, stride=2),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, 64, kernel_size=3, stride=1),
            nn.ReLU(inplace=True),
        )

        self.fc1 = nn.Linear(64 * 16 * 22, 512)
        self.act = nn.ReLU(inplace=True)
        self.fc2 = nn.Linear(512, game_inputs)
        self.optim = RMSprop(self.parameters(), lr=RMS_LEARNING_RATE,
                             momentum=RMS_GRADIENT_MOMENTUM,
                             eps=RMS_ESP)
        # more stable than mse
        self.loss_fn = nn.SmoothL1Loss()

    def forward(self, x):
        """
        Inference function
        """
        x = self.cnn(x)
        b, c, h, w = x.shape
        x = x.view(b, c * h * w)
        x = self.act(self.fc1(x))
        x = self.fc2(x)
        return x

    def backward(self, preds, expected_values):
        """
        Backwards propagation function
        """
        loss = self.loss_fn(preds, expected_values)
        wandb.log({"loss": loss})
        self.optim.zero_grad()
        loss.backward()
        nn.utils.clip_grad_value_(self.parameters(), 100)
        self.optim.step()
