import torch
import torch.nn as nn
import torch.nn.functional as F
import math

class NanoDiT(nn.Module): # 그림 그리는 학생
    """
    아주 단순화된 Diffusion Transformer (DiT) 모델
    이미지 -> 패치 -> Transformer -> 노이즈 예측
    """
    def __init__(self, img_size=32, patch_size=4, in_channels=3, embed_dim=256, depth=4, num_heads=4):
        super().__init__()
        self.img_size = img_size
        self.patch_size = patch_size
        self.num_patches = (img_size // patch_size) ** 2
        self.embed_dim = embed_dim

        # 1. Patch Embedding: 이미지를 패치 단위로 잘라 임베딩 (Conv2d가 이 역할을 잘 수행함)
        # 입력: (B, C, H, W) -> 출력: (B, Embed_dim, H/P, W/P)
        self.patch_embed = nn.Conv2d(in_channels, embed_dim, kernel_size=patch_size, stride=patch_size)

        # 2. Positional Embedding: 패치의 위치 정보 (학습 가능)
        self.pos_embed = nn.Parameter(torch.zeros(1, self.num_patches, embed_dim))

        # 3. Time Embedding: 현재 t 시점이 어디인지 알려주는 MLP
        self.time_mlp = nn.Sequential(
            nn.Linear(1, embed_dim),
            nn.GELU(),
            nn.Linear(embed_dim, embed_dim)
        )

        # 4. Transformer Encoder (Standard PyTorch Layer)
        encoder_layer = nn.TransformerEncoderLayer(d_model=embed_dim, nhead=num_heads, dim_feedforward=embed_dim*4, batch_first=True, norm_first=True)
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=depth)

        # 5. Output Head: 임베딩을 다시 이미지 픽셀(노이즈)로 복원
        self.output_head = nn.Linear(embed_dim, patch_size * patch_size * in_channels)

    def forward(self, x, t):
        """
        x: 노이즈가 섞인 이미지 (Batch, Channels, Height, Width)
        t: 시간 스텝 (Batch, 1) - 0~1 사이로 정규화된 값 권장
        """
        B, C, H, W = x.shape
        
        # --- 1. Patchify & Embedding ---
        # (B, C, H, W) -> (B, Dim, H', W') -> (B, Dim, N_patches) -> (B, N_patches, Dim)
        x = self.patch_embed(x)
        x = x.flatten(2).transpose(1, 2)

        # --- 2. Add Time & Positional Embeddings ---
        # 시간 정보를 시퀀스에 더해줌 (간단한 방식)
        t_emb = self.time_mlp(t).unsqueeze(1) # (B, 1, Dim)
        x = x + self.pos_embed + t_emb

        # --- 3. Transformer Processing ---
        x = self.transformer(x) # (B, N_patches, Dim)

        # --- 4. Unpatchify (Restore to Image) ---
        x = self.output_head(x) # (B, N_patches, Patch_size*Patch_size*C)
        
        # 다시 이미지 형태로 변환 (Reshape)
        # 이 부분은 차원 조작이 복잡하므로 주의 (B, H/P * W/P, P*P*C) -> (B, C, H, W)
        x = x.transpose(1, 2) # (B, Channels*Patch^2, N_patches)
        x = x.view(B, C, self.patch_size, self.patch_size, H // self.patch_size, W // self.patch_size)
        x = x.permute(0, 1, 4, 2, 5, 3).contiguous()
        x = x.view(B, C, H, W)
        
        return x # 예측된 노이즈

# --- Diffusion 유틸리티 (DDPM 방식 간단 구현) ---

class DiffusionTrainer: # 이미지에 먹물 뿌려버리는 선생
    # Diffusion 모델의 "노이즈 스케줄(Noise Schedule)"을 미리 계산해서 세팅해두는 부분
    # 멀쩡한 그림을 1000단계에 걸쳐서 어떻게 망가뜨릴지 계획표를 짜는 곳
    def __init__(self, model, T=1000): 
        self.model = model
        """
        self.T = 1000 (총 단계 수)
        "우리는 그림을 한 번에 망가뜨리는 게 아니라, 1000번에 걸쳐서 아주 조금씩 노이즈(먹물)를 뿌려서 망가뜨릴 거야."
        이 숫자가 클수록 그림이 더 천천히, 정교하게 변합니다.
        """
        self.T = T
        # Beta 스케줄 (Linear)
        self.beta = torch.linspace(1e-4, 0.02, T)
        self.alpha = 1. - self.beta
        self.alpha_bar = torch.cumprod(self.alpha, dim=0)

    def get_loss(self, x_0):
        """학습용: 이미지에 노이즈를 섞고, 모델이 그 노이즈를 맞추게 함"""
        B = x_0.shape[0]
        t = torch.randint(0, self.T, (B,), device=x_0.device)
        
        # 가우시안 노이즈 생성
        noise = torch.randn_like(x_0)
        
        # x_t (노이즈 낀 이미지) 생성
        a_bar = self.alpha_bar[t].view(B, 1, 1, 1).to(x_0.device)
        # 1. 멀쩡한 그림(x_0)에
        # 2. 랜덤한 먹물(noise)을 만들어서
        # 3. 정해진 비율(sqrt(a_bar))대로 섞어버림 -> x_t (문제)
        x_t = torch.sqrt(a_bar) * x_0 + torch.sqrt(1 - a_bar) * noise
        
        # 모델 예측 (시간 t는 0~1로 정규화해서 입력)
        # 학생(model)한테 "야, 내가 뿌린 먹물(noise)이 어떻게 생겼게?" 물어봄
        predicted_noise = self.model(x_t, t.view(B, 1).float() / self.T)
        
        # MSE Loss
        return F.mse_loss(predicted_noise, noise)

    @torch.no_grad()
    def sample(self, n_samples, device):
        """생성용: 랜덤 노이즈에서 시작해 점차 이미지를 복원. 노이즈 걷어내기"""
        self.model.eval()
        x = torch.randn(n_samples, 3, 32, 32).to(device) # 완전한 노이즈
        
        # loop를 돌면서 noise 조금씩 깎아냄
        for i in reversed(range(self.T)):
            t = torch.tensor([i] * n_samples, device=device).view(-1, 1).float() / self.T
            # 1. NanoDiT에게 물어봄: "야, 노이즈가 어디냐?"
            predicted_noise = self.model(x, t)
            
            # DDPM 수식에 따른 복원
            alpha = self.alpha[i].to(device)
            alpha_bar = self.alpha_bar[i].to(device)
            beta = self.beta[i].to(device)
            
            if i > 0:
                noise = torch.randn_like(x)
            else:
                noise = torch.zeros_like(x)
            
            # 2. DiffusionTrainer가 뺌 (여기가 핵심!)
            x = (1 / torch.sqrt(alpha)) * (x - ((1 - alpha) / (torch.sqrt(1 - alpha_bar))) * predicted_noise)
            x = x + torch.sqrt(beta) * noise
            
        self.model.train()
        # 값을 0~1 사이로 클리핑
        return x.clamp(-1, 1)

# --- 실행 예제 ---
if __name__ == "__main__":
    device = "cuda" if torch.cuda.is_available() else "cpu"
    
    # 1. 모델 초기화
    model = NanoDiT(img_size=32, patch_size=4, embed_dim=128, depth=4, num_heads=4).to(device)
    print(f"모델 파라미터 수: {sum(p.numel() for p in model.parameters())/1e6:.2f}M")

    # 2. Diffusion 핸들러
    diffusion = DiffusionTrainer(model)

    # 3. 가상의 학습 루프 (Dummy Data)
    optimizer = torch.optim.AdamW(model.parameters(), lr=3e-4)
    
    print("학습 시작 (Dummy Data)...")
    for epoch in range(5): # 빠르게 5번만
        # 실제로는 DataLoader에서 이미지를 가져와야 함 (값 범위 -1 ~ 1 권장)
        dummy_images = torch.randn(8, 3, 32, 32).to(device) 
        
        loss = diffusion.get_loss(dummy_images)
        
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
        print(f"Epoch {epoch+1}, Loss: {loss.item():.4f}")

    # 4. 이미지 생성 테스트
    print("이미지 생성 중...")
    generated_images = diffusion.sample(n_samples=2, device=device)
    print(f"생성 완료! 결과물 shape: {generated_images.shape}")