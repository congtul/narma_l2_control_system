from PyQt5 import QtWidgets
import os, sys
import numpy as np
import ast
import pyqtgraph as pg
import types

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from ui.advance_mode_ui import Ui_MainWindow as Ui_Advance
from backend.system_workspace import workspace


class AdvanceModeWindow(QtWidgets.QMainWindow, Ui_Advance):

    def __init__(self, parent_window=None):
        super().__init__(parent_window)
        self.setupUi(self)
        self.setWindowTitle("Advance Mode Window")
        self.parent_window = parent_window
        self.graph_win = None

        # Buttons
        self.Apply_advancemode_btn.clicked.connect(self.handle_apply)
        self.OK_advancemode_btn.clicked.connect(self.on_ok_clicked)

        self.Status_import_label.setText("Enter your generator(t) code")
        self.Status_import_label.setStyleSheet("color: gray; font-style: italic;")

    # ============================================================
    # APPLY LOGIC (MAIN)
    # ============================================================
    def handle_apply(self):
        code_text = self.GeneratorText.toPlainText().strip()

        if not code_text:
            QtWidgets.QMessageBox.warning(self, "Empty Code", "Please enter Python code first.")
            return

        is_valid, module_or_err = self.validate_python_generator(code_text)

        if not is_valid:
            QtWidgets.QMessageBox.critical(self, "Invalid Code Format", module_or_err)
            return

        module = module_or_err   # safe

        # BUILD PREVIEW
        t = np.linspace(0, workspace.run_time, int(workspace.run_time/workspace.dt) + 1)
        ref = [module.generator(tt) for tt in t]

        self.plot_graph(t, ref, title="User-defined Reference Signal (Advance Mode)")

        workspace.reference['t'] = t
        workspace.reference['ref'] = ref
        print("[INFO] User-defined reference applied to workspace.reference, runtime = ", workspace.run_time)

        self.Status_import_label.setText("Generator applied successfully")
        self.Status_import_label.setStyleSheet("color: blue; font-weight: bold;")

    # ============================================================
    # VALIDATE PYTHON STRING AS GENERATOR(t)
    # ============================================================
    def validate_python_generator(self, code_string):
        try:
            tree = ast.parse(code_string)

            generator_func = None

            for node in tree.body:

                # Only imports + function allowed
                if isinstance(node, (ast.Import, ast.ImportFrom)):
                    continue

                if isinstance(node, ast.FunctionDef) and node.name == "generator":
                    generator_func = node
                    continue

                return False, "Only imports and a generator(t) function are allowed!"

            if generator_func is None:
                return False, "Missing required function: def generator(t):"

            # Parameter check
            if len(generator_func.args.args) != 1:
                return False, "generator(t) must have exactly ONE argument"

            # Must have return
            if not any(isinstance(n, ast.Return) for n in ast.walk(generator_func)):
                return False, "generator(t) must contain a return output statement"

            # Return must return variable named 'output'
            for n in ast.walk(generator_func):
                if isinstance(n, ast.Return):
                    if not isinstance(n.value, ast.Name) or n.value.id != "output":
                        return False, "generator(t) must return a variable named 'output'"

            # SAFE EXEC: Compile code in an isolated module
            module = types.ModuleType("user_advance_mode_module")
            # Inject libraries allowed for user
            import numpy as np
            import math
            module.__dict__["np"] = np
            module.__dict__["numpy"] = np
            module.__dict__["math"] = math
            module.__dict__.update(math.__dict__)   # sin, cos, pi, sqrt...
            exec(code_string, module.__dict__)  # Safe because AST validated syntax

            # Test call
            try:
                test = module.generator(0.0)
            except Exception as e:
                return False, f"Error when executing generator(t): {e}"

            return True, module

        except Exception as e:
            return False, str(e)

    # ============================================================
    # PLOT
    # ============================================================
    def plot_graph(self, t, y, title="Preview"):
        if not self.graph_win or not self.graph_win.isVisible():
            self.graph_win = pg.PlotWidget(title=title)
            self.graph_win.setWindowTitle(title)
            self.graph_win.setBackground("#f0f0f0")
            self.graph_win.closeEvent = self.graph_close_event
            self.graph_win.show()
        else:
            self.graph_win.clear()
            self.graph_win.setTitle(title)

        pen = pg.mkPen(color='k', width=2)
        self.graph_win.plot(t, y, pen=pen)

    def graph_close_event(self, event):
        self.graph_win = None
        event.accept()

    # ============================================================
    # OK BUTTON
    # ============================================================
    def on_ok_clicked(self):
        if self.Apply_advancemode_btn.isEnabled():
            # chưa apply → tự động apply
            self.handle_apply()
        self.close()

    def closeEvent(self, event):
        if self.graph_win and self.graph_win.isVisible():
            self.graph_win.close()
            self.graph_win = None

        if self.parent_window:
            self.parent_window.show()

        event.accept()



if __name__ == "__main__":
    app = QtWidgets.QApplication(sys.argv)
    win = AdvanceModeWindow()
    win.show()
    sys.exit(app.exec_())
