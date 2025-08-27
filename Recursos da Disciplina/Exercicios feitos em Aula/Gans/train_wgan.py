# Treinamento WGAN-GP para MNIST (célula / script pronto)
import os
import time
import torch
from torch import optim

# Imports dos seus scripts
from data_setup import create_dataloaders        # retorna train_dataloader. :contentReference[oaicite:3]{index=3}
from model import GeneratorMLP, DiscriminatorMLP  # usa as arquiteturas MLP para estabilidade. :contentReference[oaicite:4]{index=4}
from utils import _gradient_penalty, save_10_images, print_train_time  # gp, salvar imagens, relatório de tempo. :contentReference[oaicite:5]{index=5}

# -----------------------
# Hiperparâmetros
# -----------------------
DATA_DIR = "./data"
BATCH_SIZE = 64
NUM_WORKERS = 4
EPOCHS = 10
NOISE_DIM = 100
LR = 1e-4
BETA1, BETA2 = 0.0, 0.9  # recomendado para WGAN-GP
N_CRITIC = 5             # atualizações do descriminador por passo do gerador
LAMBDA_GP = 10.0
SAMPLE_INTERVAL = 500    # salvar imagens a cada N iterações (ajuste conforme quiser)
CHECKPOINT_DIR = "checkpoints"
os.makedirs("training", exist_ok=True)
os.makedirs(CHECKPOINT_DIR, exist_ok=True)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Device:", device)

# -----------------------
# Preparar dados e modelos
# -----------------------
train_loader = create_dataloaders(DATA_DIR, BATCH_SIZE, NUM_WORKERS)  # :contentReference[oaicite:6]{index=6}

gen = GeneratorMLP().to(device)         # saída em (-1,1) por Tanh. :contentReference[oaicite:7]{index=7}
disc = DiscriminatorMLP().to(device)    # produz escalar por exemplo (batch,1) → view(-1). :contentReference[oaicite:8]{index=8}

opt_gen = optim.Adam(gen.parameters(), lr=LR, betas=(BETA1, BETA2))
opt_disc = optim.Adam(disc.parameters(), lr=LR, betas=(BETA1, BETA2))

# fix seed (opcional)
torch.manual_seed(42)

# -----------------------
# Funções utilitárias
# -----------------------
def sample_noise(batch_size, dim, device):
    return torch.randn(batch_size, dim, device=device)

def save_checkpoint(epoch, it):
    path_g = os.path.join(CHECKPOINT_DIR, f"generator_epoch{epoch}_it{it}.pt")
    path_d = os.path.join(CHECKPOINT_DIR, f"discriminator_epoch{epoch}_it{it}.pt")
    torch.save(gen.state_dict(), path_g)
    torch.save(disc.state_dict(), path_d)

# -----------------------
# Loop de Treino
# -----------------------
start_time = time.time()
iter_count = 0

for epoch in range(1, EPOCHS + 1):
    for real_imgs, _ in train_loader:
        iter_count += 1
        real_imgs = real_imgs.to(device)

        # normalizar imagens reais para [-1, 1] (generator usa tanh)
        real_imgs = real_imgs * 2.0 - 1.0

        # -----------------------
        # Treinar o Discriminador / Crítico N_CRITIC vezes (reaproveitando o mesmo batch)
        # Critic loss: E[D(fake)] - E[D(real)] + lambda * gp
        # -----------------------
        for _ in range(N_CRITIC):
            opt_disc.zero_grad()

            batch_size = real_imgs.size(0)
            z = sample_noise(batch_size, NOISE_DIM, device)
            fake_imgs = gen(z)  # (batch,1,28,28) - saída do GeneratorMLP. :contentReference[oaicite:9]{index=9}

            # outputs do crítico (como vetor)
            d_real = disc(real_imgs).view(-1)
            d_fake = disc(fake_imgs.detach()).view(-1)

            # gradient penalty (usa função do seu utils). :contentReference[oaicite:10]{index=10}
            gp = _gradient_penalty(disc, real_imgs, fake_imgs, device)

            loss_disc = d_fake.mean() - d_real.mean() + LAMBDA_GP * gp
            loss_disc.backward()
            opt_disc.step()

        # -----------------------
        # Treinar o Gerador
        # Gen loss: - E[D(fake)]
        # -----------------------
        opt_gen.zero_grad()
        z = sample_noise(batch_size, NOISE_DIM, device)
        fake_imgs = gen(z)
        d_fake_for_gen = disc(fake_imgs).view(-1)
        loss_gen = -d_fake_for_gen.mean()
        loss_gen.backward()
        opt_gen.step()

        # Logs simples
        if iter_count % 50 == 0:
            print(f"Epoch [{epoch}/{EPOCHS}] Iter {iter_count}  Loss_D: {loss_disc.item():.4f}  Loss_G: {loss_gen.item():.4f}  GP: {gp.item():.4f}")

        # Salva amostras e checkpoints
        if iter_count % SAMPLE_INTERVAL == 0:
            # salvar 20 imagens (sua função de utilidade cria uma grade 4x5 e espera múltiplos)
            imgs_to_save = fake_imgs[:20]
            save_10_images(imgs_to_save, iter_count)  # :contentReference[oaicite:11]{index=11}
            save_checkpoint(epoch, iter_count)
            print(f"Saved samples & checkpoint at iter {iter_count}")

    # fim do epoch: salvar checkpoint por epoch
    save_checkpoint(epoch, iter_count)
    print(f"Epoch {epoch} completo. Checkpoints salvos.")

end_time = time.time()
print_train_time(start_time, end_time, device=device)  # usa sua função util. :contentReference[oaicite:12]{index=12}
