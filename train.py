import torch
import wandb
from tqdm import tqdm
from data_loader import get_dataloaders
from simclr_model import SimCLR, nt_xent_loss
from dinov2.models import dinov2_vits14

DATA_PATH = "/scratch/yj3076/content/data/train"

def train():
    wandb.init(project="final_ssl_project", name="simclr_resnet_dinov2")

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print("Using device:", device)

    loader = get_dataloaders(DATA_PATH, batch_size=128)

    model = SimCLR()
    model = model.to(device)

    dinov2 = dinov2_vits14(pretrained=True).to(device)
    dinov2.eval()

    optimizer = torch.optim.Adam(model.parameters(), lr=3e-4)

    for epoch in range(10):
        total_loss = 0
        
        for images, _ in tqdm(loader):
            images = images.to(device)

            _, z1 = model(images)
            with torch.no_grad():
                dino_out = dinov2(images)

            _, z2 = model(images)

            loss = nt_xent_loss(z1, z2)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            total_loss += loss.item()

        wandb.log({"epoch": epoch, "loss": total_loss / len(loader)})

        print(f"Epoch {epoch}: Loss = {total_loss / len(loader)}")

    wandb.finish()


if __name__ == "__main__":
    train()
