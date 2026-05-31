import os
import torch
import torch.nn as nn
from torchvision import transforms
from PIL import Image
from torch.utils.data import Dataset, DataLoader

# ===== PATH SETUP =====
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
real_path = os.path.join(BASE_DIR, "data", "raw", "real")

# ===== TRANSFORM =====
transform = transforms.Compose([
    transforms.Resize((128, 128)),
    transforms.ToTensor()
])

# ===== DATASET =====
class RealDataset(Dataset):
    def __init__(self, folder):
        self.paths = [os.path.join(folder, f) for f in os.listdir(folder)]
    
    def __len__(self):
        return len(self.paths)
    
    def __getitem__(self, idx):
        img = Image.open(self.paths[idx]).convert("RGB")
        return transform(img)

dataset = RealDataset(real_path)
loader = DataLoader(dataset, batch_size=8, shuffle=True)

# ===== MODEL (AUTOENCODER) =====
class Autoencoder(nn.Module):
    def __init__(self):
        super().__init__()
        
        self.encoder = nn.Sequential(
            nn.Conv2d(3, 16, 3, stride=2, padding=1),  # 128 → 64
            nn.ReLU(),
            nn.Conv2d(16, 32, 3, stride=2, padding=1), # 64 → 32
            nn.ReLU()
        )
        
        self.decoder = nn.Sequential(
            nn.ConvTranspose2d(32, 16, 2, stride=2),   # 32 → 64
            nn.ReLU(),
            nn.ConvTranspose2d(16, 3, 2, stride=2),    # 64 → 128
            nn.Sigmoid()
        )
    
    def forward(self, x):
        x = self.encoder(x)
        x = self.decoder(x)
        return x

model = Autoencoder()
criterion = nn.MSELoss()
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

# ===== TRAINING =====
epochs = 111

for epoch in range(epochs):
    total_loss = 0
    
    for imgs in loader:
        outputs = model(imgs)
        loss = criterion(outputs, imgs)
        
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
        total_loss += loss.item()
    
    print(f"Epoch {epoch+1}, Loss: {total_loss:.4f}")

# ===== SAVE MODEL =====
model_path = os.path.join(BASE_DIR, "models", "autoencoder.pth")
torch.save(model.state_dict(), model_path)

print("Model saved successfully!")