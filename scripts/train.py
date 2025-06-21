import torch
import torch.nn as nn
import torch.optim as optim
from tqdm import tqdm
import sys
import os

# Add parent directory to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from config.config import Config
from data.dataset import create_dataloader
from models.steganography_model import stegGNN
from utils.training import adjust_learning_rate, forward_pass, save_checkpoint, load_checkpoint

def train(model, dataloader, config, ckpt_dir="checkpoints"):
    """Main training function with checkpoint directory support."""
    # Ensure checkpoint directory exists
    os.makedirs(ckpt_dir, exist_ok=True)

    criterion = nn.MSELoss().to(config.device)
    optimizer = optim.Adam(
        model.parameters(),
        lr=config.initial_lr,
        betas=config.optimizer_betas,
        eps=1e-6,
        weight_decay=config.weight_decay
    )
    model = model.to(config.device).train()

    # Paths for continuing and best model
    last_ckpt = os.path.join(ckpt_dir, "last_checkpoint.pt")
    best_ckpt = os.path.join(ckpt_dir, "best_model.pt")

    epoch_start = 0
    best_loss = float('inf')

    # Load last checkpoint if available
    if os.path.exists(last_ckpt):
        print(f"Loading last checkpoint from {last_ckpt}...")
        epoch_start, loaded_loss = load_checkpoint(model, optimizer, last_ckpt)
        print(f"Resuming from epoch {epoch_start+1}, loss={loaded_loss:.4f}")
        best_loss = loaded_loss if os.path.exists(best_ckpt) else best_loss

    # Load best model's loss
    if os.path.exists(best_ckpt):
        try:
            best_data = torch.load(best_ckpt, map_location=config.device)
            best_loss = best_data.get('best_loss', best_loss)
            print(f"Previous best loss: {best_loss:.4f}")
        except Exception:
            print("Could not read best_model.pt; starting with default best_loss")

    for epoch in range(epoch_start, config.num_epochs):
        adjust_learning_rate(optimizer, epoch, config)

        if epoch == epoch_start:
            print(f"Training started ({config.num_epochs} epochs) from epoch {epoch_start+1}")

        progress_bar = tqdm(dataloader, desc=f"Epoch {epoch+1}/{config.num_epochs}")
        running_loss = 0.0
        hiding_loss = 0.0
        revealing_loss = 0.0

        for batch_idx, (image, _) in enumerate(progress_bar):
            secret_img = image[image.shape[0]//2:].to(config.device)
            cover_img = image[:image.shape[0]//2].to(config.device)

            optimizer.zero_grad()
            loss, h_err, r_err = forward_pass(secret_img, cover_img, model, criterion, config)
            loss.backward()
            optimizer.step()

            running_loss += loss.item()
            hiding_loss += h_err.item()
            revealing_loss += r_err.item()

            # Format loss to 4 decimal places correctly
            formatted_loss = running_loss / (batch_idx + 1)
            progress_bar.set_postfix({'loss': f"{formatted_loss:.4f}"})

        # Calculate average losses
        avg_loss = running_loss / len(dataloader)
        avg_hiding = hiding_loss / len(dataloader)
        avg_revealing = revealing_loss / len(dataloader)
        print(f"Epoch [{epoch+1}/{config.num_epochs}], Loss: {avg_loss:.4f}, "
              f"Hiding_loss: {avg_hiding:.4f}, Revealing_loss: {avg_revealing:.4f}")

        # Save last checkpoint every epoch
        save_checkpoint(model, optimizer, epoch, avg_loss, last_ckpt)

        # Save best model
        if avg_loss < best_loss:
            best_loss = avg_loss
            save_checkpoint(model, optimizer, epoch, best_loss, best_ckpt)
            print(f"New best model saved with loss: {best_loss:.4f}")

if __name__ == "__main__":
    config = Config()
    model = stegGNN(config.img_size, config.in_channels, config.embedding_dim)
    train_loader = create_dataloader(config)
    train(model, train_loader, config)
