from PyQt5 import QtWidgets
import os, sys
import numpy as np
import ast
import pyqtgraph as pg
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from ui.advance_mode_ui import Ui_MainWindow as Ui_Advance

class AdvanceModeWindow(QtWidgets.QMainWindow, Ui_Advance):
    saved_file_path = None

    def __init__(self, parent_window=None):
        super().__init__(parent_window)
        self.setupUi(self)
        self.setWindowTitle("Advance Mode Window")
        self.parent_window = parent_window
        self.graph_win = None

        # Buttons
        self.Import_btn.clicked.connect(self.import_file)
        self.Apply_advancemode_btn.clicked.connect(self.apply_imported_file)
        self.Clear_code_btn.clicked.connect(self.clear_imported_file)
        self.OK_advancemode_btn.clicked.connect(self.on_ok_clicked)

        if AdvanceModeWindow.saved_file_path:
            self.imported_file = AdvanceModeWindow.saved_file_path
            self.Status_import_label.setText(
                f"Applied File: {os.path.basename(self.imported_file)}")
            self.Status_import_label.setStyleSheet("color: green; font-weight: bold;")
            self.Apply_advancemode_btn.setEnabled(False)
        else:
            self.imported_file = None
            self.Status_import_label.setText("No file imported")
            self.Status_import_label.setStyleSheet("color: gray; font-style: italic;")
            self.Apply_advancemode_btn.setEnabled(False)

    def import_file(self):
        file_path, _ = QtWidgets.QFileDialog.getOpenFileName(
            self, "Import Source Code", "", "Python Files (*.py);;All Files (*)")
        if not file_path:
            self.Status_import_label.setText("Import cancelled")
            self.Status_import_label.setStyleSheet("color: gray; font-style: italic;")
            return
        if file_path.endswith(".py"):
            self.imported_file = file_path
            self.Status_import_label.setText(f"File imported: {os.path.basename(file_path)}")
            self.Status_import_label.setStyleSheet("color: green; font-weight: bold;")
            if file_path != AdvanceModeWindow.saved_file_path:
                self.Apply_advancemode_btn.setEnabled(True)
            else:
                self.Apply_advancemode_btn.setEnabled(False)
        else:
            QtWidgets.QMessageBox.critical(
                self, "Invalid File", "Please import a Python (.py) file only."
            )

    def apply_imported_file(self):
        if not self.imported_file:
            QtWidgets.QMessageBox.warning(
                self, "No File", "No file has been imported yet.")
            return
        
        #check python is valid or not
        is_valid, result = self.validate_python_generator(self.imported_file)

        if not is_valid:
            QtWidgets.QMessageBox.critical(self, "Invalid Python Format", result)
            return
        else:
            # Save imported module for later use
            self.loaded_module = result
            AdvanceModeWindow.saved_file_path = self.imported_file

        #get runtime_input from main_window
        runtime_text = self.main_window_ref.Run_time_input.text().strip()

        if not runtime_text:
            QtWidgets.QMessageBox.warning(
                self,
                "Missing Runtime",
                "Please enter a run time in the Main Window before applying the Advance Mode file."
            )
            return

        try:
            t_runtime = float(runtime_text)
            if t_runtime <= 0:
                raise ValueError
        except ValueError:
            QtWidgets.QMessageBox.critical(
                self,
                "Invalid Runtime",
                "Run time must be a positive number."
            )
            return

        print("Debug: t_runtime =", t_runtime)  
        
        t = np.linspace(0, t_runtime, 300)

        y = [self.loaded_module.generator(val) for val in t]

        # Show graph using pyqtgraph
        self.plot_graph(t, y, title="User-defined Reference Signal (Advance Mode)")
        self.Status_import_label.setText(f"Input values are set ({os.path.basename(self.imported_file)})")
        self.Status_import_label.setStyleSheet("color: blue; font-weight: bold;")
        self.Apply_advancemode_btn.setEnabled(False)

        return self.loaded_module.generator(t_runtime)
    
    def plot_graph(self, t, y, title="Preview"):
        if not self.graph_win or not self.graph_win.isVisible():
            # Tạo PlotWidget lần đầu hoặc khi cửa sổ bị đóng
            self.graph_win = pg.PlotWidget(title=title)
            self.graph_win.setWindowTitle(title)
            self.graph_win.setBackground("#f0f0f0")  # nền xám/trắng

            # Bắt sự kiện đóng cửa sổ
            self.graph_win.closeEvent = self.graph_close_event

            self.graph_win.show()
        else:
            # Clear graph nếu còn sống
            self.graph_win.clear()
            self.graph_win.setTitle(title)

        pen = pg.mkPen(color='k', width=2)  # đường màu đen
        self.graph_win.plot(t, y, pen=pen)
    
    def graph_close_event(self, event):
        # Khi user đóng cửa sổ graph, set graph_win = None để Apply có thể tạo lại
        self.graph_win = None
        event.accept()

    def validate_python_generator(self, file_path):
        try:
            # Read full file
            with open(file_path, "r", encoding="utf-8") as f:
                code = f.read()

            # Parse AST (no execution)
            tree = ast.parse(code)

            generator_func = None

            # Find function named generator
            for node in tree.body:
                # Only allow imports, comments, function definition
                if isinstance(node, ast.Import) or isinstance(node, ast.ImportFrom):
                    continue

                if isinstance(node, ast.FunctionDef) and node.name == "generator":
                    generator_func = node
                    continue

                # Anything else at top-level is FORBIDDEN
                return False, "Only imports and generator(t) function are allowed at top level!"

            if generator_func is None:
                return False, "Missing required function: def generator(t):"

            # Check function must have exactly 1 parameter
            if len(generator_func.args.args) != 1:
                return False, "generator(t) must have exactly ONE argument"

            # Check function has return statement
            has_return = any(isinstance(n, ast.Return) for n in ast.walk(generator_func))
            if not has_return:
                return False, "generator(t) must contain a return output statement"

            # OPTIONAL: Check returned variable is named output
            for n in ast.walk(generator_func):
                if isinstance(n, ast.Return):
                    if not isinstance(n.value, ast.Name) or n.value.id != "output":
                        return False, "generator(t) must return variable named 'output'"

            # After format validation, NOW we safely import
            import importlib.util
            spec = importlib.util.spec_from_file_location("user_code", file_path)
            module = importlib.util.module_from_spec(spec)
            spec.loader.exec_module(module)

            # Test call
            try:
                test = module.generator(0.0)
            except Exception as e:
                return False, f"Error when executing generator(t): {e}"

            return True, module

        except Exception as e:
            return False, str(e)


    def clear_imported_file(self):
        self.imported_file = None
        AdvanceModeWindow.saved_file_path = None
        self.Status_import_label.setText("All values cleared")
        self.Status_import_label.setStyleSheet("color: red; font-weight: bold;")
        self.Apply_advancemode_btn.setEnabled(False)

    def on_ok_clicked(self):
        if self.imported_file and self.Apply_advancemode_btn.isEnabled():
            self.apply_imported_file()
        self.close()

    def closeEvent(self, event):
        if self.graph_win and self.graph_win.isVisible():
            self.graph_win.close()
            self.graph_win = None
            
        if self.parent_window:
            self.parent_window.show()
        event.accept()

if __name__ == "__main__":
    import sys
    app = QtWidgets.QApplication(sys.argv)
    win = AdvanceModeWindow()
    win.show()
    sys.exit(app.exec_())