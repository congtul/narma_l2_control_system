Here are the steps required to set up and run the project. Please follow the instructions below in order to run the system.
1. **Get the Source Code**\
Get the source code of the project from our group submission or directly from GitHub.
2. **Install Necessary Packages**\
This project is developed and tested using Python 3.11.6.
It is recommended that Python 3.11.6 is installed on your system.
Then install necessary packages by running: “pip install -r requirements.txt”.
3. **Start The Project**\
Start the project by running: “python main.py”.
4. **System Workflow Overview**\
The workflow of our system follows a strict sequence to ensure that the NARMA-L2 model is trained and used correctly for control. The steps are summarized as follows:
    1. **Login to The System:**\
Users need to be authenticated before using the system. Our system has 2 types of user: admin user and normal user, each of them plays a different role in how the NARMA-L2 controller is developed, configured, and used.\
For fully access and control of the system, please user the account with admin privilege:\
username: user1\
password: 123
    2. **Define the Plant Model:**\
The first requirement is to define the plant model in the Plant Model block. Without this step, no dataset can be generated and the NARMA-L2 model cannot be trained, since the model must learn from the plant’s input–output behavior.
    3. **Configure the NARMA-L2 Model:**\
After defining the plant, the user sets up the NARMA-L2 model by selecting nu, ny , and the hidden layer size. The user must also specify the minimum and maximum values of inputs and outputs (e.g., voltage range and motor speed range). Note that ny and nu do not strictly depend on the true order of G(z); in practice, even when the system order is unknown, the NARMA-L2 network can still learn to approximate the plant.
    4. **Dataset Preparation:**\
The software provides multiple ways to prepare training data: importing an external dataset or generating one internally. In all cases, the dataset is obtained by applying an input sequence to the plant model and recording the corresponding output. Therefore, dataset generation must occur after the plant model is defined.
    5. **Training the NARMA-L2 Model:**\
The model is trained using the generated dataset. While batch size and learning rate are fixed in this version of the software, the user can choose the number of epochs and the early stopping learning. Visualization tools are provided to help users evaluate whether the model has achieved acceptable prediction accuracy. Importantly, the trained model is not applied to the controller until the user explicitly confirms it by clicking Save Model.
    6. **Reference Input Definition:**\
Once a reliable predictive model is obtained, it is used as a controller. The user defines the reference trajectory (motion planning), and the NARMA-L2 controller computes the control input so the plant output can track the reference.
    7. **Simulation with Training Options:**\
The system supports both offline training (before simulation) and online training during simulation, allowing the controller to adapt or refine its understanding of the plant in real time.
    8. **Performance Evaluation:**\
The software provides visualization tools for evaluating both prediction performance and tracking performance. Additional numerical indicators are also computed to help assess the effectiveness of the NARMA-L2 model.
If you want more information about the workflows of our system, please refer to section 4: NARMA-L2 Controller Training and section 5: Controller Performance and Evaluation. These sections are the detailed versions of the above general workflow.
