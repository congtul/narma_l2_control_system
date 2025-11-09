# narma_l2_control_system
project/
│── ui/                         # Chứa UI file (.ui) và các file convert .py
│   ├── main_window_ui.py
│   ├── input_config_window_ui.py
│   ├── model_params_window_ui.py
│   ├── visualization_window_ui.py
│   ├── simulation_panel_window_ui.py
│   └── resources_rc.py         # Convert từ resources.qrc
│
│── windows/                     # Các class cửa sổ PyQt, xử lý slot/event
│   ├── main_window.py
│   ├── input_config_window.py
│   ├── model_params_window.py
│   ├── visualization_window.py
│   └── simulation_panel_window.py
│
│── control_sys/                       # Xử lý thuật toán, controller, plotting
│   ├── narma_model.py           # Thuật toán NARMA-L2
│   ├── controller.py            # Logic điều khiển, gọi model
│   └── plot_engine.py           # Matplotlib embed vào PyQt
│
│── resources/                   # Lưu file resource như ảnh, icon, style
│   ├── images/
│   │   ├── logo.png
│   │   ├── icon_play.png
│   │   └── ...
│   └── styles/
│       └── style.qss
│
│── main.py                       # Entry point: chạy QApplication
│── requirements.txt              # Thư viện cần cài
└── README.md
