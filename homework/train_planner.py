"""
Usage:
    python3 -m homework.train_planner --your_args here
"""
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
            '''
            train_preds = train_results.argmax(dim=1)
            train_acc = (train_preds == label).float().mean()
            metrics["train_acc"].append(train_acc)
            '''

            metrics["train_loss"].append(train_losses.item())

            #raise NotImplementedError("Training step not implemented")

            global_step += 1

        # torch.inference_mode calls model.eval() and disables gradient computation
        with torch.inference_mode():
            for img, label in val_data:
                img, label = img.to(device), label.to(device)

                
                val_results = model.forward(img)
                
                # ChatGPT 4o-mini
                '''
                val_preds = val_results.argmax(dim=1)
                val_acc = (val_preds == label).float().mean()
                metrics["val_acc"].append(val_acc)
                '''
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

print("Time to train")
