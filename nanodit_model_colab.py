import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision
import torchvision.transforms as transforms
from torch.utils.data import DataLoader, Subset
import matplotlib.pyplot as plt
import numpy as np

# ==========================================
# 1. 모델 정의 (NanoDiT) - 가볍게 설정
# ==========================================
class NanoDiT(nn.Module):
    def __init__(self, img_size=32, patch_size=4, in_channels=3, embed_dim=128, depth=6, num_heads=4):
        super().__init__()
        self.patch_size = patch_size
        self.num_patches = (img_size // patch_size) ** 2
        
        self.patch_embed = nn.Conv2d(in_channels, embed_dim, kernel_size=patch_size, stride=patch_size)
        self.pos_embed = nn.Parameter(torch.zeros(1, self.num_patches, embed_dim))
        
        self.time_mlp = nn.Sequential(
            nn.Linear(1, embed_dim),
            nn.GELU(),
            nn.Linear(embed_dim, embed_dim)
        )
        
        encoder_layer = nn.TransformerEncoderLayer(d_model=embed_dim, nhead=num_heads, 
                                                   dim_feedforward=embed_dim*2, 
                                                   dropout=0.1, batch_first=True, norm_first=True)
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=depth)
        self.output_head = nn.Linear(embed_dim, patch_size * patch_size * in_channels)

    def forward(self, x, t):
        B, C, H, W = x.shape
        x = self.patch_embed(x).flatten(2).transpose(1, 2)
        t_emb = self.time_mlp(t).unsqueeze(1)
        x = x + self.pos_embed + t_emb
        x = self.transformer(x)
        x = self.output_head(x)
        x = x.transpose(1, 2).view(B, C, self.patch_size, self.patch_size, H // self.patch_size, W // self.patch_size)
        x = x.permute(0, 1, 4, 2, 5, 3).contiguous().view(B, C, H, W)
        return x

# ==========================================
# 2. Diffusion 학습 도구 (Scheduler)
# ==========================================
class DiffusionTrainer:
    def __init__(self, model, device, T=1000):
        self.model = model
        self.device = device
        self.T = T
        self.beta = torch.linspace(1e-4, 0.02, T).to(device)
        self.alpha = 1. - self.beta
        self.alpha_bar = torch.cumprod(self.alpha, dim=0)

    def get_loss(self, x_0):
        B = x_0.shape[0]
        t = torch.randint(0, self.T, (B,), device=self.device)
        noise = torch.randn_like(x_0)
        
        a_bar = self.alpha_bar[t].view(B, 1, 1, 1)
        x_t = torch.sqrt(a_bar) * x_0 + torch.sqrt(1 - a_bar) * noise
        
        # 입력 t를 0~1로 스케일링
        predicted_noise = self.model(x_t, t.view(B, 1).float() / self.T)
        return F.mse_loss(predicted_noise, noise)

    @torch.no_grad()
    def sample(self, n_samples):
        self.model.eval()
        x = torch.randn(n_samples, 3, 32, 32).to(self.device)
        print(f"이미지 생성 시작 (Steps: {self.T})...")
        
        for i in reversed(range(self.T)):
            t = torch.tensor([i] * n_samples, device=self.device).view(-1, 1).float() / self.T
            predicted_noise = self.model(x, t)
            
            alpha = self.alpha[i]
            alpha_bar = self.alpha_bar[i]
            beta = self.beta[i]
            
            if i > 0:
                noise = torch.randn_like(x)
            else:
                noise = torch.zeros_like(x)
            
            x = (1 / torch.sqrt(alpha)) * (x - ((1 - alpha) / (torch.sqrt(1 - alpha_bar))) * predicted_noise)
            x = x + torch.sqrt(beta) * noise
            
        self.model.train()
        return x.clamp(-1, 1)

# ==========================================
# 3. 실행 및 학습 (개구리만 골라서 학습)
# ==========================================
def main():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"사용 장치: {device}")
    
    # 데이터셋 설정 (CIFAR-10)
    transform = transforms.Compose([
        transforms.RandomHorizontalFlip(), # 데이터 증강
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)) # -1 ~ 1 로 정규화
    ])
    
    print("데이터셋 다운로드 중...")
    dataset = torchvision.datasets.CIFAR10(root='./data', train=True, download=True, transform=transform)
    
    # 중요: 개구리(Class index 6)만 골라내기!
    frog_indices = [i for i, label in enumerate(dataset.targets) if label == 6]
    frog_dataset = Subset(dataset, frog_indices)
    
    dataloader = DataLoader(frog_dataset, batch_size=64, shuffle=True, num_workers=2)
    print(f"학습용 개구리 이미지 수: {len(frog_dataset)}장")

    # 모델 생성
    model = NanoDiT().to(device)
    trainer = DiffusionTrainer(model, device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=5e-4) # 학습률 중요
    
    # 학습 루프 (약 1시간 목표 -> 60 Epoch 정도 예상)
    epochs = 50 
    print(f"학습 시작! (총 {epochs} Epochs)")
    
    loss_history = []
    
    for epoch in range(epochs):
        epoch_loss = 0
        for images, _ in dataloader:
            images = images.to(device)
            loss = trainer.get_loss(images)
            
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            epoch_loss += loss.item()
        
        avg_loss = epoch_loss / len(dataloader)
        loss_history.append(avg_loss)
        print(f"Epoch [{epoch+1}/{epochs}] Loss: {avg_loss:.4f}")
        
        # 중간 점검 (10 에포크마다)
        if (epoch + 1) % 10 == 0:
             print(f"-- {epoch+1} 에포크 경과, 손실값 줄어드는 중 --")

    print("학습 완료. 이미지 생성 중...")
    
    # 결과 확인
    generated_imgs = trainer.sample(n_samples=16) # 16마리 생성
    
    # 시각화
    generated_imgs = (generated_imgs + 1) / 2 # -1~1 -> 0~1 로 복구
    grid_img = torchvision.utils.make_grid(generated_imgs, nrow=4).cpu().permute(1, 2, 0).numpy()
    
    plt.figure(figsize=(8, 8))
    plt.imshow(grid_img)
    plt.axis('off')
    plt.title("Generated Frogs (NanoDiT)")
    plt.show()

if __name__ == "__main__":
    main()