from PyQt5 import QtWidgets, QtCore
import os

class UserGuideWindow(QtWidgets.QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("User Guide")
        self.resize(900, 600)
        self.browser = QtWidgets.QTextBrowser()
        script_dir = os.path.dirname(os.path.abspath(__file__))
        guide_path = os.path.join(script_dir, '..', 'resources', 'docs', 'user_guide.html')
        guide_path = os.path.abspath(guide_path)
        if os.path.exists(guide_path):
            self.browser.setSource(QtCore.QUrl.fromLocalFile(guide_path))
        else:
            self.browser.setText("<h2 style='color:red;'>User guide file not found.</h2>")
        self.setCentralWidget(self.browser)

if __name__ == "__main__":
    import sys
    app = QtWidgets.QApplication(sys.argv)
    win = UserGuideWindow()
    win.show()
    sys.exit(app.exec_())