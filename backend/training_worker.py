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
        model = self.controller.model
        ny = model.ny
        nu = model.nu
        lr = self.controller.lr
        epochs = self.controller.epochs
        batch_size = self.controller.batch_size

        optimizer = optim.Adam(model.parameters(), lr=lr)
        loss_fn = nn.MSELoss()

        train_loader = DataLoader(self.train_ds, batch_size=batch_size, shuffle=True)
        val_loader = DataLoader(self.val_ds, batch_size=batch_size, shuffle=False) if self.val_ds else None

        history = []

        for epoch in range(1, epochs + 1):

            if self.stopped:
                return {"status": "stopped", "history": history}

            # ---------------------- TRAIN ----------------------
            model.train()
            train_loss = 0
            for X, Y, _ in train_loader:
                optimizer.zero_grad()
                pred = model(X)
                loss = loss_fn(pred, Y)
                loss.backward()
                optimizer.step()
                train_loss += loss.item()

            train_loss /= max(1, len(train_loader))

            # ---------------------- VAL ----------------------
            val_loss = None
            if val_loader:
                model.eval()
                with torch.no_grad():
                    val_loss = 0
                    for Xv, Yv, _ in val_loader:
                        pv = model(Xv)
                        lv = loss_fn(pv, Yv)
                        val_loss += lv.item()
                    val_loss /= max(1, len(val_loader))

            history.append((epoch, train_loss, val_loss))

            # Emit về GUI
            self.epoch_signal.emit(epoch, train_loss, val_loss)

        return {"status": "ok", "history": history}
