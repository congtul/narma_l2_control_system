from PyQt5 import QtWidgets, QtCore
from PyQt5.QtCore import Qt
import os, sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from ui.main_ui import Ui_MainWindow as Ui_Main
from ui.input_ref_ui import Ui_MainWindow as Ui_Input
from ui.normal_mode_ui import Ui_MainWindow as Ui_Normal
from ui.advance_mode_ui import Ui_MainWindow as Ui_Advance


# === Window classes ===
class InputWindow(QtWidgets.QMainWindow, Ui_Input):
    def __init__(self):
        super().__init__()
        self.setupUi(self)
        self.setWindowTitle("Input Window")

        # install double-click filters for its own buttons
        self.Normal_mode_btn.installEventFilter(self)
        self.Advance_mode_btn.installEventFilter(self)

    def eventFilter(self, obj, event):
        if event.type() == event.MouseButtonDblClick and event.button() == Qt.LeftButton:
            if obj == self.Normal_mode_btn:
                self.open_normal_mode()
            elif obj == self.Advance_mode_btn:
                self.open_advance_mode()
            return True
        return super().eventFilter(obj, event)

    def open_normal_mode(self):
        self.normal_window = NormalModeWindow(self)
        self.normal_window.show()
        self.hide()  # hide instead of close, so we can reopen later

    def open_advance_mode(self):
        self.advance_window = AdvanceModeWindow(self)
        self.advance_window.show()
        self.hide()


# === Sub-windows ===
class NormalModeWindow(QtWidgets.QMainWindow, Ui_Normal):
    # Persistent storage for last applied values
    saved_mode = None
    saved_random_values = {"max_vel": "", "min_vel": ""}
    saved_manual_values = {"stime": "", "veloval": ""}
    def __init__(self, parent_window=None):
        super().__init__()
        self.setupUi(self)
        self.setWindowTitle("Normal Mode Window")
        self.parent_window = parent_window  # store InputWindow reference

        restored = self.restore_previous_state()
        if not restored:
            # Initially disable all editable fields
            self.RD_max_vel_line_edit.setEnabled(False)
            self.RD_min_vel_line_edit.setEnabled(False)
            self.man_Stime_lineedit.setEnabled(False)
            self.man_Veloval_lineedit.setEnabled(False)

        
        # Connect buttons to actions
        self.Random_gen_btn.clicked.connect(self.use_random_mode)
        self.Manual_gen_btn.clicked.connect(self.use_manual_mode)
        self.Reset_all_val_btn.clicked.connect(self.reset_all_values)
        self.Apply_normalmode_btn.clicked.connect(self.apply_normal_mode)
        self.OK_normalmode_btn_2.clicked.connect(self.on_ok_clicked)

        # Connect text change signals to update Apply button state
        self.RD_max_vel_line_edit.textChanged.connect(self.update_apply_button_state)
        self.RD_min_vel_line_edit.textChanged.connect(self.update_apply_button_state)
        self.man_Stime_lineedit.textChanged.connect(self.update_apply_button_state)
        self.man_Veloval_lineedit.textChanged.connect(self.update_apply_button_state)

        if not restored:
        # Track which mode is active
            self.active_mode = None

        #print(f"{self.active_mode} values applied successfully!")

    def restore_previous_state(self):
        # Load previously saved state
        if NormalModeWindow.saved_mode == "random":
            self.active_mode = "random"
            self.use_random_mode()
            self.RD_max_vel_line_edit.setText(NormalModeWindow.saved_random_values["max_vel"]) # get save_values
            self.RD_min_vel_line_edit.setText(NormalModeWindow.saved_random_values["min_vel"])
            self.man_Stime_lineedit.setText(NormalModeWindow.saved_manual_values["stime"])
            self.man_Veloval_lineedit.setText(NormalModeWindow.saved_manual_values["veloval"])
            self.RD_max_vel_line_edit.setEnabled(True)
            self.RD_min_vel_line_edit.setEnabled(True)
            self.man_Stime_lineedit.setEnabled(False)
            self.man_Veloval_lineedit.setEnabled(False)
            
            return True

        elif NormalModeWindow.saved_mode == "manual":
            self.active_mode = "manual"
            self.use_manual_mode()
            self.man_Stime_lineedit.setText(NormalModeWindow.saved_manual_values["stime"]) #get saved_values
            self.man_Veloval_lineedit.setText(NormalModeWindow.saved_manual_values["veloval"])
            self.RD_max_vel_line_edit.setText(NormalModeWindow.saved_random_values["max_vel"])
            self.RD_min_vel_line_edit.setText(NormalModeWindow.saved_random_values["min_vel"])
            self.man_Stime_lineedit.setEnabled(True)
            self.man_Veloval_lineedit.setEnabled(True)
            self.RD_max_vel_line_edit.setEnabled(False)
            self.RD_min_vel_line_edit.setEnabled(False)
            return True

        else:
            self.active_mode = None
            self.Apply_normalmode_btn.setEnabled(False) # Apply button disabled until a new change happens
            return False
    
    def use_random_mode(self):

        self.active_mode = "random"

        # Label feedback
        self.Status_nor_label.setText("Random mode is chosen!")
        self.Status_nor_label.setStyleSheet("color: #ffa23a; font-weight: bold;")

        # Enable random inputs
        self.RD_max_vel_line_edit.setEnabled(True)
        self.RD_min_vel_line_edit.setEnabled(True)

        # Disable manual inputs and button
        self.man_Stime_lineedit.setEnabled(False)
        self.man_Veloval_lineedit.setEnabled(False)

        # Update button state
        self.update_apply_button_state()
   
    def use_manual_mode(self):

        self.active_mode = "manual"

        # Label feedback
        self.Status_nor_label.setText("Manual mode is chosen!")
        self.Status_nor_label.setStyleSheet("color: blue; font-weight: bold;")

        # Enable manual inputs
        self.man_Stime_lineedit.setEnabled(True)
        self.man_Veloval_lineedit.setEnabled(True)

        # Disable random inputs and button
        self.RD_max_vel_line_edit.setEnabled(False)
        self.RD_min_vel_line_edit.setEnabled(False)

        # Update button state
        self.update_apply_button_state()
        
    def reset_all_values(self):

        self.active_mode = None

        # Clear all QLineEdit fields
        self.RD_max_vel_line_edit.clear()
        self.RD_min_vel_line_edit.clear()
        self.man_Stime_lineedit.clear()
        self.man_Veloval_lineedit.clear()

        # Disable all input fields again
        self.RD_max_vel_line_edit.setEnabled(False)
        self.RD_min_vel_line_edit.setEnabled(False)
        self.man_Stime_lineedit.setEnabled(False)
        self.man_Veloval_lineedit.setEnabled(False)

        # Disable Apply button
        self.Apply_normalmode_btn.setEnabled(False)

        # Update label
        self.Status_nor_label.setText("Reset all values")
        self.Status_nor_label.setStyleSheet("color: red; font-weight: bold;")

# ==================== Apply Button Logic ====================

    def update_apply_button_state(self):
        """Enable Apply button only if proper fields are filled for the current mode."""
        #print(f"{self.active_mode} values applied successfully!")
        if self.active_mode == "random":
            if (self.RD_max_vel_line_edit.text().strip() and
                self.RD_min_vel_line_edit.text().strip()):
                self.Apply_normalmode_btn.setEnabled(True)
            else:
                self.Apply_normalmode_btn.setEnabled(False)

        elif self.active_mode == "manual":
            if (self.man_Stime_lineedit.text().strip() and
                self.man_Veloval_lineedit.text().strip()):
                self.Apply_normalmode_btn.setEnabled(True)
            else:
                self.Apply_normalmode_btn.setEnabled(False)

        else:
            self.Apply_normalmode_btn.setEnabled(False)


    def apply_normal_mode(self):
        """Handle Apply button click depending on mode."""
        if self.active_mode == "random":
            max_vel = self.RD_max_vel_line_edit.text()  #### Example to get value
            min_vel = self.RD_min_vel_line_edit.text()

            # Save persistent values
            NormalModeWindow.saved_mode = "random"
            NormalModeWindow.saved_random_values = {
                "max_vel": max_vel,
                "min_vel": min_vel,
            }

            print(f"[Random Mode Applied] Max Vel = {max_vel}, Min Vel = {min_vel}")

        elif self.active_mode == "manual":
            stime = self.man_Stime_lineedit.text()      #### Example to get value
            veloval = self.man_Veloval_lineedit.text()

            # Save persistent values
            NormalModeWindow.saved_mode = "manual"
            NormalModeWindow.saved_manual_values = {
                "stime": stime,
                "veloval": veloval,
            }

            print(f"[Manual Mode Applied] Start Time = {stime}, Velocity = {veloval}")

        else:
            print("No mode active!")

        self.Status_nor_label.setText(f"{self.active_mode.capitalize()} values applied successfully!")
        self.Status_nor_label.setStyleSheet("color: green; font-weight: bold;")

        # Disable Apply button after applying
        self.Apply_normalmode_btn.setEnabled(False)
    
    def on_ok_clicked(self):
        """Handle OK button — simply close the window after ensuring state is applied."""
        # Optional: auto-apply before closing if Apply wasn't pressed
        if self.Apply_normalmode_btn.isEnabled():
            self.apply_normal_mode()

        # Close the current window
        self.close()

    def closeEvent(self, event):
        """When this window closes, reopen the InputWindow."""
        if self.parent_window:
            self.parent_window.show()
        event.accept()


class AdvanceModeWindow(QtWidgets.QMainWindow, Ui_Advance):
    # Persistent storage for applied file
    saved_file_path = None

    def __init__(self, parent_window=None):
        super().__init__()
        self.setupUi(self)
        self.setWindowTitle("Advance Mode Window")
        self.parent_window = parent_window

        # === Connect Buttons ===
        self.Import_btn.clicked.connect(self.import_file)
        self.Apply_advancemode_btn.clicked.connect(self.apply_imported_file)
        self.Clear_code_btn.clicked.connect(self.clear_imported_file)
        self.OK_advancemode_btn.clicked.connect(self.on_ok_clicked)

        # === Restore previous file if available ===
        if AdvanceModeWindow.saved_file_path:
            self.imported_file = AdvanceModeWindow.saved_file_path
            self.Status_import_label.setText(
                f"Applied File: {os.path.basename(self.imported_file)}"
            )
            self.Status_import_label.setStyleSheet("color: green; font-weight: bold;")
            self.Apply_advancemode_btn.setEnabled(False)  # already applied
        else:
            self.imported_file = None
            self.Status_import_label.setText("No file imported")
            self.Status_import_label.setStyleSheet("color: gray; font-style: italic;")
            self.Apply_advancemode_btn.setEnabled(False)

    # === Import file logic ===
    def import_file(self):
        """Open a file dialog to select a Python or C/C++ file."""
        while True:
            file_path, _ = QtWidgets.QFileDialog.getOpenFileName(
                self,
                "Import Source Code",
                "",
                "Source Files (*.py *.c *.cpp);;All Files (*)"
            )

            if not file_path:
                # User cancelled
                self.Status_import_label.setText("Import cancelled")
                self.Status_import_label.setStyleSheet("color: gray; font-style: italic;")
                return

            # Validate extension
            if file_path.endswith((".py", ".c", ".cpp")):
                self.imported_file = file_path
                filename = os.path.basename(file_path)
                self.Status_import_label.setText(f"File imported: {filename}")
                self.Status_import_label.setStyleSheet("color: green; font-weight: bold;")
                print(f"[File Imported] {file_path}")

                # Enable Apply only if new file or changed file
                if file_path != AdvanceModeWindow.saved_file_path:
                    self.Apply_advancemode_btn.setEnabled(True)
                else:
                    self.Apply_advancemode_btn.setEnabled(False)
                break
            else:
                QtWidgets.QMessageBox.critical(
                    self,
                    "Invalid File",
                    "Invalid file type! Please import a Python (.py), C (.c), or C++ (.cpp) file."
                )
                # Reopen dialog automatically (loop continues)

    # === Apply button logic ===
    def apply_imported_file(self):
        """Confirm the imported file and persist its path."""
        if not self.imported_file:    ##warning if there is no file imported but applied button is pressed 
            QtWidgets.QMessageBox.warning(
                self,
                "No File",
                "No file has been imported yet. Please import a file first."
            )
            return

        # Save persistently
        AdvanceModeWindow.saved_file_path = self.imported_file
        filename = os.path.basename(self.imported_file)
        self.Status_import_label.setText(f"Input values are set ({filename})")
        self.Status_import_label.setStyleSheet("color: blue; font-weight: bold;")
        print(f"[Applied File] {self.imported_file}")

        # Disable Apply until user imports again
        self.Apply_advancemode_btn.setEnabled(False)

    # === Clear all ===
    def clear_imported_file(self):
        """Remove imported file and reset state."""
        self.imported_file = None
        AdvanceModeWindow.saved_file_path = None

        self.Status_import_label.setText("All values cleared")
        self.Status_import_label.setStyleSheet("color: red; font-weight: bold;")
        self.Apply_advancemode_btn.setEnabled(False)

        print("[Advance Mode] File cleared successfully")

    # === OK button ===
    def on_ok_clicked(self):
        """Close window safely after ensuring state is applied."""
        if self.imported_file and self.Apply_advancemode_btn.isEnabled():
            # Auto-apply if imported but not applied yet
            self.apply_imported_file()

        self.close()

    # === Close event ===
    def closeEvent(self, event):
        """When this window closes, reopen the InputWindow."""
        if self.parent_window:
            self.parent_window.show()
        event.accept()

class UserGuideWindow(QtWidgets.QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("User Guide")
        self.resize(900, 600)

        # QTextBrowser can display HTML content
        self.browser = QtWidgets.QTextBrowser()
        script_dir = os.path.dirname(os.path.abspath(__file__))
        guide_path = os.path.join(script_dir, '..', 'resources', 'docs', 'user_guide.html')
        guide_path = os.path.abspath(guide_path)
        print(f"Loading user guide from: {guide_path}")

        if os.path.exists(guide_path):
            self.browser.setSource(QtCore.QUrl.fromLocalFile(guide_path))
        else:
            self.browser.setText("<h2 style='color:red;'>User guide file not found.</h2>")

        # Layout
        self.setCentralWidget(self.browser)

# === Main window logic ===
class MainApp(QtWidgets.QMainWindow, Ui_Main):
    def __init__(self):
        super().__init__()
        self.setupUi(self)
        self.setWindowTitle("Main Window")

        # install event filters for double-clicks
        for btn in [
            self.input_btn,
            self.ANN_controller_btn,
            self.DC_motor_btn,
            self.Output_btn,
            self.Run_btn,
            self.User_guide_btn,
        ]:
            btn.installEventFilter(self)    

    def eventFilter(self, obj, event):
        if event.type() == event.MouseButtonDblClick and event.button() == Qt.LeftButton:
            if obj == self.input_btn:
                self.open_input_window()
            elif obj == self.ANN_controller_btn:
                self.open_ann_controller_window()
            elif obj == self.DC_motor_btn:
                self.open_dc_motor_window()
            elif obj == self.Output_btn:
                self.open_output_window()
            elif obj == self.Run_btn:
                self.open_run_window()
            elif obj == self.User_guide_btn:
                self.open_user_guide_window()
            return True
        return super().eventFilter(obj, event)

    def closeEvent(self, event):
        """When main window closes, close all other open windows."""
        QtWidgets.QApplication.closeAllWindows()
        event.accept()

    # === Window actions ===
    def open_input_window(self):
        self.input_window = InputWindow()
        self.input_window.show()

    def open_ann_controller_window(self):
        print("Double-click ANN Controller → open ANN window here")

    def open_dc_motor_window(self):
        print("Double-click DC Motor → open DC Motor window here")

    def open_output_window(self):
        print("Double-click Output → open Output window here")

    def open_run_window(self):
        print("Double-click RUN → open RUN window here")

    def open_user_guide_window(self):
        """Open the User Guide window."""
        self.user_guide_window = UserGuideWindow()
        self.user_guide_window.show()
