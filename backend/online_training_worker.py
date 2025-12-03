from PyQt5.QtCore import QObject
import torch
import time
from backend.system_workspace import workspace

class OnlineTrainingWorker(QObject):
    def __init__(self, model, lr=5e-5, batch_size=5, epoch=2):
        super().__init__()
        self.model = model
        if workspace.training_online:
            self.optimizer = torch.optim.Adam(self.model.parameters(), lr=lr)
        self.loss_fn = torch.nn.MSELoss()

        # --- NARMA histories ---
        self.y_hist = torch.zeros(model.ny)
        self.u_hist = torch.zeros(model.nu)

        # --- training config ---
        self.batch_size = batch_size
        self.epoch = epoch
        self.timeout_ms = workspace.dt * 1000 * batch_size  # ms

        # --- batch buffer ---
        self.batch = []

        self.running = True

    # Simulation push 1 sample mới
    def push_sample(self, yk_1, u, y_real):
        self.y_hist = torch.roll(self.y_hist, 1)
        self.y_hist[0] = yk_1

        self.batch.append((
            self.y_hist.clone(),
            self.u_hist.clone(),
            torch.tensor([u], dtype=torch.float32),
            torch.tensor([y_real], dtype=torch.float32)
        ))

        self.u_hist = torch.roll(self.u_hist, 1)
        self.u_hist[0] = u

    # Worker vòng lặp train
    def run(self):
        while self.running:
            if not workspace.training_online:
                time.sleep(0.1)
                continue

            if len(self.batch) >= self.batch_size:
                start_time = time.perf_counter()

                for i in range(self.epoch):
                    for y_hist, u_hist, u_tensor, y_real in self.batch:
                        # timeout check
                        elapsed_ms = (time.perf_counter() - start_time) * 1000
                        if elapsed_ms > self.timeout_ms:
                            print(f"⏱ Epoch {i+1} timeout after {elapsed_ms:.2f} ms")
                            break

                        inp = torch.cat([y_hist, u_hist])
                        y_pred = self.model.f(inp).squeeze() + self.model.g(inp).squeeze() * u_tensor.squeeze()

                        y_real = y_real.squeeze()
                        loss = self.loss_fn(y_pred, y_real)
                        self.optimizer.zero_grad()
                        loss.backward()
                        self.optimizer.step()

                    else:
                        continue
                    break  # timeout → thoát epoch

                self.batch.clear()
            else:
                time.sleep(self.timeout_ms / 1000)

    def stop(self):
        self.running = False
