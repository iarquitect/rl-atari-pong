"""Agente DQN que aprende a jugar Pong (Atari) desde los píxeles.

Arquitectura de Mnih et al. (2015), "Human-level control through deep
reinforcement learning", Nature 518. Se puede cortar y retomar: cada tanda
guarda el modelo y el epsilon en `cerebro_pong.pth` y la siguiente sigue
desde ahí.

Uso:
    python entrenar_pong.py                  # 50 partidas, video cada 10
    python entrenar_pong.py --partidas 200 --video-cada 50
"""
import argparse
import os
import random
from collections import deque

import ale_py
import gymnasium as gym
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from gymnasium.wrappers import AtariPreprocessing, FrameStackObservation, RecordVideo

# Hiperparámetros
GAMMA = 0.99
BATCH_SIZE = 32
BUFFER_CAPACITY = 10_000  # el paper usa 1 millón; con 10 mil entra en la RAM de una laptop
LEARNING_RATE = 1e-4
SYNC_TARGET_STEPS = 1_000
EPSILON_DECAY = 0.995  # por partida
EPSILON_MIN = 0.05
GUARDAR_CADA = 25  # partidas
CHECKPOINT_FILE = "cerebro_pong.pth"


class AtariDQN(nn.Module):
    """Tres convoluciones y dos capas densas: entra la pila de 4 cuadros de 84x84,
    sale un valor Q por cada acción posible."""

    def __init__(self, n_actions):
        super().__init__()
        self.conv1 = nn.Conv2d(4, 32, kernel_size=8, stride=4)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=4, stride=2)
        self.conv3 = nn.Conv2d(64, 64, kernel_size=3, stride=1)
        self.fc1 = nn.Linear(64 * 7 * 7, 512)
        self.fc2 = nn.Linear(512, n_actions)

    def forward(self, x):
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        x = F.relu(self.conv3(x))
        x = x.view(x.size(0), -1)
        x = F.relu(self.fc1(x))
        return self.fc2(x)


class ReplayBuffer:
    """Memoria de jugadas. Muestrear al azar rompe la correlación entre cuadros seguidos."""

    def __init__(self, capacity):
        self.buffer = deque(maxlen=capacity)

    def push(self, state, action, reward, next_state, terminated):
        self.buffer.append((state, action, reward, next_state, terminated))

    def sample(self, batch_size):
        state, action, reward, next_state, terminated = zip(*random.sample(self.buffer, batch_size))
        return np.array(state), action, reward, np.array(next_state), terminated

    def __len__(self):
        return len(self.buffer)


def crear_entorno(video_cada):
    gym.register_envs(ale_py)
    env = gym.make("ALE/Pong-v5", render_mode="rgb_array")
    # La cámara va antes del preprocesado para grabar la imagen original a color
    env = RecordVideo(
        env,
        video_folder="./video_pong_local",
        episode_trigger=lambda episode_id: episode_id % video_cada == 0,
    )
    # Pong-v5 ya saltea 4 cuadros por su cuenta, por eso acá frame_skip=1
    env = AtariPreprocessing(env, screen_size=84, grayscale_obs=True, frame_skip=1, noop_max=30)
    return FrameStackObservation(env, 4)


def select_action(env, model, state_tensor, epsilon, device):
    if random.random() < epsilon:
        return env.action_space.sample()  # exploración
    with torch.no_grad():
        return model(state_tensor.to(device)).argmax().item()  # explotación


def compute_td_loss(model, target_model, optimizer, replay_buffer, device):
    state, action, reward, next_state, terminated = replay_buffer.sample(BATCH_SIZE)

    state = torch.FloatTensor(np.float32(state)).to(device)
    next_state = torch.FloatTensor(np.float32(next_state)).to(device)
    action = torch.LongTensor(action).to(device)
    reward = torch.FloatTensor(reward).to(device)
    terminated = torch.FloatTensor(terminated).to(device)

    q_value = model(state).gather(1, action.unsqueeze(1)).squeeze(1)
    next_q_value = target_model(next_state).max(1)[0]

    # Ecuación de Bellman. Solo un final real del partido corta el valor futuro;
    # un corte por límite de tiempo no, porque el partido podría haber seguido.
    expected_q_value = reward + GAMMA * next_q_value * (1 - terminated)
    loss = F.smooth_l1_loss(q_value, expected_q_value.detach())

    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
    return loss


def guardar(model, epsilon):
    torch.save({"model_state_dict": model.state_dict(), "epsilon": epsilon}, CHECKPOINT_FILE)


def main():
    parser = argparse.ArgumentParser(description="Entrena un agente DQN en Pong.")
    parser.add_argument("--partidas", type=int, default=50, help="partidas de esta tanda (50 por defecto)")
    parser.add_argument("--video-cada", type=int, default=10, help="grabar una partida cada N (10 por defecto)")
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Iniciando entrenamiento usando: {device}")

    env = crear_entorno(args.video_cada)
    model = AtariDQN(env.action_space.n).to(device)
    target_model = AtariDQN(env.action_space.n).to(device)
    optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)
    replay_buffer = ReplayBuffer(BUFFER_CAPACITY)

    if os.path.exists(CHECKPOINT_FILE):
        print(f"Se encontró un cerebro previo ({CHECKPOINT_FILE}). Cargando conocimientos...")
        checkpoint = torch.load(CHECKPOINT_FILE, map_location=device, weights_only=True)
        model.load_state_dict(checkpoint["model_state_dict"])
        epsilon = checkpoint["epsilon"]
        print(f"Resumiendo con un Epsilon de: {epsilon:.3f}")
    else:
        print("No se encontraron guardados previos. Iniciando de cero...")
        epsilon = 1.0
    target_model.load_state_dict(model.state_dict())

    print(f"¡Arrancan las {args.partidas} partidas!")
    total_steps = 0
    for episode in range(args.partidas):
        state, info = env.reset()
        done = False
        episode_reward = 0

        while not done:
            state_tensor = torch.FloatTensor(np.array(state)).unsqueeze(0)
            action = select_action(env, model, state_tensor, epsilon, device)
            next_state, reward, terminated, truncated, info = env.step(action)
            done = terminated or truncated

            replay_buffer.push(state, action, reward, next_state, terminated)
            state = next_state
            episode_reward += reward
            total_steps += 1

            if len(replay_buffer) > BATCH_SIZE:
                compute_td_loss(model, target_model, optimizer, replay_buffer, device)

            if total_steps % SYNC_TARGET_STEPS == 0:
                target_model.load_state_dict(model.state_dict())

        epsilon = max(EPSILON_MIN, epsilon * EPSILON_DECAY)
        print(
            f"Episodio {episode + 1}/{args.partidas} | Recompensa: {episode_reward} | "
            f"Pasos de la sesión: {total_steps} | Epsilon: {epsilon:.3f}"
        )

        if (episode + 1) % GUARDAR_CADA == 0 or episode + 1 == args.partidas:
            print(f"Guardando checkpoint en el episodio {episode + 1}...")
            guardar(model, epsilon)

    env.close()
    print("¡Entrenamiento finalizado! Volvé a correr el script para seguir aprendiendo.")


if __name__ == "__main__":
    main()
