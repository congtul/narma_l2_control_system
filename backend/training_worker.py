# training_worker.py
from PyQt5 import QtCore
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader


class TrainingWorker(QtCore.QThread):
    """
    QThread chuyên train NARMA-L2
    - Phát tín hiệu mỗi epoch để GUI cập nhật real-time
    - Phát tín hiệu khi train xong
    """

    epoch_signal = QtCore.pyqtSignal(int, float, float)    # epoch, train_loss, val_loss
    finished_signal = QtCore.pyqtSignal(object)            # result dict
    stopped = False

    def __init__(self, controller, train_ds, val_ds=None, parent=None):
        super().__init__(parent)
        self.controller = controller
        self.train_ds = train_ds
        self.val_ds = val_ds

    def stop(self):
        self.stopped = True

    def run(self):
        try:
            result = self._train_narma()
            self.finished_signal.emit(result)

        except Exception as e:
            self.finished_signal.emit({"status": "error", "message": str(e)})

    # ---------------------------------------------------

    def _train_narma(self):
        device = (torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu"))
        self.controller.f.to(device)
        self.controller.g.to(device)
        train_loader = DataLoader(self.train_ds, batch_size=self.controller.batch_size, shuffle=False)
        val_loader = DataLoader(self.val_ds, batch_size=self.controller.batch_size, shuffle=False) if self.val_ds else None
        params = list(self.controller.f.parameters()) + list(self.controller.g.parameters())
        optim = torch.optim.Adam(params, lr=self.controller.lr)
        loss_fn = nn.MSELoss()

        best_val_loss = float("inf")
        patience_counter = 0
        best_state = None

        history = []

        for epoch in range(1, self.controller.epochs + 1):
            
            if self.stopped:
                return {"status": "stopped", "history": history}

            # ---------------------- TRAIN ----------------------
            self.controller.f.train()
            self.controller.g.train()
            train_loss_acc = 0.0
            n_train = 0

            for batch in train_loader:
                # support both (x,y,u) and (x,y)
                if len(batch) == 3:
                    x, y_real, u_k = batch
                else:
                    x, y_real = batch
                    # try to extract u from last nu columns of x
                    u_k = x[:, -self.controller.nu]

                x = x.to(device); y_real = y_real.to(device); u_k = u_k.to(device)
                
                optim.zero_grad()
                y_pred = self.controller.f(x).squeeze() + self.controller.g(x).squeeze() * u_k.squeeze()
                loss = loss_fn(y_pred, y_real.view_as(y_pred))
                loss.backward()
                optim.step()

                batch_n = x.size(0)
                train_loss_acc += loss.item() * batch_n
                n_train += batch_n

            train_loss = train_loss_acc / max(1, n_train)

            # ---------------------- VAL ----------------------
            val_loss = None
            if val_loader:
                self.controller.f.eval()
                self.controller.g.eval()
                val_loss_acc = 0.0
                n_val = 0
                with torch.no_grad():
                    for batch in val_loader:
                        if len(batch) == 3:
                            x, y_real, u_k = batch
                        else:
                            x, y_real = batch
                            u_k = x[:, -self.controller.nu]
                        x = x.to(device); y_real = y_real.to(device); u_k = u_k.to(device)
                        y_pred = self.controller.f(x).squeeze() + self.controller.g(x).squeeze() * u_k.squeeze()
                        loss = loss_fn(y_pred, y_real.view_as(y_pred))
                        batch_n = x.size(0)
                        val_loss_acc += loss.item() * batch_n
                        n_val += batch_n
                val_loss = val_loss_acc / max(1, n_val)

                if val_loss < best_val_loss:
                    best_val_loss = val_loss
                    patience_counter = 0
                    best_state = {
                        "f": {k: v.cpu() for k, v in self.controller.f.state_dict().items()},
                        "g": {k: v.cpu() for k, v in self.controller.g.state_dict().items()},
                    }
                else:
                    patience_counter += 1
                    if patience_counter >= self.controller.patience:
                        print("Early stopping triggered.")
                        break

            history.append((epoch, train_loss, val_loss))

            # Emit về GUI
            self.epoch_signal.emit(epoch, train_loss, val_loss)

        # Load best weights if available
        if best_state is not None:
            self.controller.f.load_state_dict(best_state["f"])
            self.controller.g.load_state_dict(best_state["g"])

        return {"status": "ok", "history": history}
