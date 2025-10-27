"""
Usage:
    python3 -m homework.train_planner --your_args here
"""
"""
Usage:
    python3 -m homework.train_planner --model_name mlp_planner --num_epoch 50 --lr 1e-3
"""

#Claude Sonnet 4.5
import argparse
from datetime import datetime
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.utils.tensorboard as tb

from .models import load_model, save_model
from .datasets.road_dataset import load_data

# ChatGPT 4.0-mini
class RMSELoss(nn.Module):
    def __init__(self):
        super().__init__()
        self.mse = nn.MSELoss()

    def forward(self, y_pred, y_true):
        return torch.sqrt(self.mse(y_pred, y_true))


def train(
    exp_dir: str = "logs",
    model_name: str = "mlp_planner",
    num_epoch: int = 50,
    lr: float = 1e-3,
    batch_size: int = 128,
    seed: int = 2024,
    **kwargs,
):
    if torch.cuda.is_available():
        device = torch.device("cuda")
    else:
        print("CUDA not available, using CPU")
        device = torch.device("cpu")

    # set random seed so each run is deterministic
    torch.manual_seed(seed)
    np.random.seed(seed)

    # directory with timestamp to save tensorboard logs and model checkpoints
    log_dir = Path(exp_dir) / f"{model_name}_{datetime.now().strftime('%m%d_%H%M%S')}"
    logger = tb.SummaryWriter(log_dir)

    # note: the grader uses default kwargs, you'll have to bake them in for the final submission
    model = load_model(model_name, **kwargs)
    model = model.to(device)
    model.train()

    # Load road data with track boundaries and waypoints
    train_data = load_data(
        "drive_data/train", 
        shuffle=True, 
        batch_size=batch_size, 
        num_workers=2
    )
    val_data = load_data(
        "drive_data/val", 
        shuffle=False, 
        batch_size=batch_size, 
        num_workers=2
    )

    # create loss function and optimizer
    loss_func = RMSELoss() # sqrt to get RMSE if this flops
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)

    global_step = 0

    # training loop
    for epoch in range(num_epoch):
        model.train()
        metrics = {"train_loss": []}

        for batch in train_data:
            # Get track boundaries and ground truth waypoints
            track_left = batch['track_left'].to(device)
            track_right = batch['track_right'].to(device)
            waypoints_gt = batch['waypoints'].to(device)
            waypoints_mask = batch['waypoints_mask'].to(device)
            
            # Forward pass
            waypoints_pred = model(track_left=track_left, track_right=track_right)
            
            # Calculate loss (only on valid waypoints)
            # Expand mask to match waypoint dimensions: (B, n_waypoints) -> (B, n_waypoints, 2)
            mask = waypoints_mask.unsqueeze(-1).expand_as(waypoints_pred)
            
            # Compute loss only on masked (valid) waypoints
            loss = loss_func(waypoints_pred[mask], waypoints_gt[mask])
            
            # Backward pass
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            metrics["train_loss"].append(loss.item())
            global_step += 1

        # Validation
        model.eval()
        val_metrics = {"val_loss": [], "longitudinal_error": [], "lateral_error": []}
        
        with torch.inference_mode():
            for batch in val_data:
                track_left = batch['track_left'].to(device)
                track_right = batch['track_right'].to(device)
                waypoints_gt = batch['waypoints'].to(device)
                waypoints_mask = batch['waypoints_mask'].to(device)
                
                # Forward pass
                waypoints_pred = model(track_left=track_left, track_right=track_right)
                
                # Calculate loss
                mask = waypoints_mask.unsqueeze(-1).expand_as(waypoints_pred)
                val_loss = loss_func(waypoints_pred[mask], waypoints_gt[mask])
                val_metrics["val_loss"].append(val_loss.item())
                
                # Calculate longitudinal error (forward direction, x-axis)
                long_error = torch.abs(waypoints_pred[..., 0] - waypoints_gt[..., 0])
                long_error = long_error[waypoints_mask].mean()
                val_metrics["longitudinal_error"].append(long_error.item())
                
                # Calculate lateral error (left/right direction, y-axis)
                lat_error = torch.abs(waypoints_pred[..., 1] - waypoints_gt[..., 1])
                lat_error = lat_error[waypoints_mask].mean()
                val_metrics["lateral_error"].append(lat_error.item())

        # Log metrics to tensorboard
        epoch_train_loss = torch.as_tensor(metrics["train_loss"]).mean()
        epoch_val_loss = torch.as_tensor(val_metrics["val_loss"]).mean()
        epoch_long_error = torch.as_tensor(val_metrics["longitudinal_error"]).mean()
        epoch_lat_error = torch.as_tensor(val_metrics["lateral_error"]).mean()

        logger.add_scalar("train_loss", epoch_train_loss, global_step)
        logger.add_scalar("val_loss", epoch_val_loss, global_step)
        logger.add_scalar("longitudinal_error", epoch_long_error, global_step)
        logger.add_scalar("lateral_error", epoch_lat_error, global_step)

        # Print progress on first, last, and every 10th epoch
        if epoch == 0 or epoch == num_epoch - 1 or (epoch + 1) % 10 == 0:
            print(
                f"Epoch {epoch + 1:2d} / {num_epoch:2d}: "
                f"train_loss={epoch_train_loss:.4f} "
                f"val_loss={epoch_val_loss:.4f} "
                f"long_error={epoch_long_error:.4f} "
                f"lat_error={epoch_lat_error:.4f}"
            )

    # Save model for grading
    save_model(model)
    
    # Save a copy in the log directory
    torch.save(model.state_dict(), log_dir / f"{model_name}.th")
    print(f"\nTraining complete!")
    print(f"Model saved to {log_dir / f'{model_name}.th'}")
    print(f"Final metrics:")
    print(f"  Longitudinal error: {epoch_long_error:.4f}")
    print(f"  Lateral error: {epoch_lat_error:.4f}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train a planner model")

    parser.add_argument("--exp_dir", type=str, default="logs", help="Directory for logs")
    parser.add_argument("--model_name", type=str, required=True, 
                        choices=["mlp_planner", "transformer_planner", "vit_planner"],
                        help="Model to train")
    parser.add_argument("--num_epoch", type=int, default=50, help="Number of epochs")
    parser.add_argument("--lr", type=float, default=1e-3, help="Learning rate")
    parser.add_argument("--batch_size", type=int, default=128, help="Batch size")
    parser.add_argument("--seed", type=int, default=2024, help="Random seed")

    # Optional: additional model hyperparameters
    # parser.add_argument("--hidden_size", type=int, default=256)
    # parser.add_argument("--num_layers", type=int, default=3)

    # Pass all arguments to train
    args = parser.parse_args()
    train(**vars(args))

'''
import argparse
from datetime import datetime
from pathlib import Path

import numpy as np
import torch
import torch.utils.tensorboard as tb

import torch.optim

from .models import load_model, save_model
from .utils import load_data


def train(
    exp_dir: str = "logs",
    model_name: str = "linear",
    num_epoch: int = 50,
    lr: float = 1e-3,
    batch_size: int = 128,
    seed: int = 2024,
    **kwargs,
):
    if torch.cuda.is_available():
        device = torch.device("cuda")
    else:
        print("CUDA not available, using CPU")
        device = torch.device("cpu")

    # set random seed so each run is deterministic
    torch.manual_seed(seed)
    np.random.seed(seed)

    # directory with timestamp to save tensorboard logs and model checkpoints
    log_dir = Path(exp_dir) / f"{model_name}_{datetime.now().strftime('%m%d_%H%M%S')}"
    logger = tb.SummaryWriter(log_dir)

    # note: the grader uses default kwargs, you'll have to bake them in for the final submission
    model = load_model(model_name, **kwargs)
    model = model.to(device)
    model.train()

    train_data = load_data("data/train", shuffle=True, batch_size=batch_size, num_workers=2)
    val_data = load_data("data/val", shuffle=False)

    # create loss function and optimizer
    loss_func = nn.RMSELoss()
    optimizer = torch.optim.SGD(model.parameters(), lr, momentum=0.9)

    global_step = 0


    # training loop
    for epoch in range(num_epoch):
        metrics = {"train_loss": [], "val_loss": []}

        for img, label in train_data:
            img, label = img.to(device), label.to(device)

            # pseudocode for steps come form ChatGPT 4o-mini
            train_results = model.forward(img) # move forward
            train_losses = loss_func.forward(train_results, label) # calculate loss
            

            # ChatGPT 4o-mini
            optimizer.zero_grad()     # 1. clear old gradients
            train_losses.backward()   # 2. compute gradients
            optimizer.step()          # 3. update weights

            # checking accuracy for this epoch - ChatGPT 4o-mini


            metrics["train_loss"].append(train_losses.item())

            #raise NotImplementedError("Training step not implemented")

            global_step += 1

        # torch.inference_mode calls model.eval() and disables gradient computation
        with torch.inference_mode():
            for img, label in val_data:
                img, label = img.to(device), label.to(device)

                
                val_results = model.forward(img)
                
                # ChatGPT 4o-mini
                val_loss = loss_func(val_results, label).item()
                metrics["val_loss"].append(val_loss)

        # log average train and val accuracy to tensorboard
        epoch_train_loss = torch.as_tensor(metrics["train_loss"]).mean()
        epoch_val_loss = torch.as_tensor(metrics["val_loss"]).mean()

        logger.add_scalar("train_loss", epoch_train_loss, global_step)
        logger.add_scalar("val_loss", epoch_val_loss, global_step)

        # print on first, last, every 10th epoch
        if epoch == 0 or epoch == num_epoch - 1 or (epoch + 1) % 10 == 0:
            print(
                f"Epoch {epoch + 1:2d} / {num_epoch:2d}: "
                f"train_acc={epoch_train_acc:.4f} "
                f"val_acc={epoch_val_acc:.4f}"
            )


    print('################################################################3')
    # save and overwrite the model in the root directory for grading
    save_model(model)
    print('--------------------------------------------------------------------')
    # save a copy of model weights in the log directory
    torch.save(model.state_dict(), log_dir / f"{model_name}.th")
    print(f"Model saved to {log_dir / f'{model_name}.th'}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument("--exp_dir", type=str, default="logs")
    parser.add_argument("--model_name", type=str, required=True)
    parser.add_argument("--num_epoch", type=int, default=50)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--seed", type=int, default=2024)


    # optional: additional model hyperparamters
    # parser.add_argument("--num_layers", type=int, default=3)

    # pass all arguments to train
    train(**vars(parser.parse_args()))
'''
print("Time to train")
