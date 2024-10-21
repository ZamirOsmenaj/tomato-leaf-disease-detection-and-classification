# From Pixels to Diagnosis: Machine Learning Approaches for Tomato Leaf Disease Detection

## Tomato Leaf Disease Detection and Classification

### Description

The prevalence of diseases in tomato plants poses a significant challenge to agriculture, necessitating innovative solutions to address this problem. This project, part of a thesis, explores the application of Machine Learning (ML) and Artificial Intelligence (AI) to develop a model capable of identifying and classifying diseases in tomato leaves. 

The project involves:

- Developing a custom Convolutional Neural Network (CNN) trained on a diverse dataset of tomato leaf images.
- Fine-tuning pre-trained models like VGG16 and VGG19 for comparative analysis.
- Creating a user-friendly web application for real-world use, allowing users to upload tomato leaf images and receive disease diagnoses and treatment recommendations.

This research highlights the effectiveness of ML and AI in agricultural disease management and opens up possibilities for real-time and mobile platform integration.

---

### Table of Contents
1. [Prerequisites](#prerequisites)
1. [Installation](#installation)
1. [Usage](#usage)
1. [Project Structure](#project-structure)
1. [Contributing](#contributing)
1. [License](#license)

---

### Prerequisites

Before starting, ensure you have the following installed:

- **Python 3.10**: You can download it from the [official Python website](https://www.python.org/downloads/).
- **pip** (Python package installer): It usually comes with Python, but if not, you can download it using:
  ```bash
  curl https://bootstrap.pypa.io/get-pip.py -o get-pip.py
  python3.10 get-pip.py
  ```

Make sure Python and pip are installed correctly by running:

```bash
python3.10 --version
pip3 --version
```

For GPU support and improved performance, it's recommended to have **CUDA** and **cuDNN** installed on your system. If you're using a GPU, please refer to the full thesis document for further instructions on setting up TensorFlow for GPU usage.

---

### Installation

Follow these steps to set up the project on your local machine:

1. **Clone the Repository**  
   Download the `Thesis` repository from GitHub:
   ```bash
   git clone https://github.com/ZamirOsmenaj/Tomato-Leaf-Disease-Detection-and-Classification
   ```

1. **Set Up a Virtual Environment**  
   Create a virtual environment (replace `{venv_name}` with the name of your choice):
   ```bash
   python3.10 -m venv {venv_name}
   ```

1. **Activate the Virtual Environment**  
   On Linux/macOS:
   ```bash
   source {venv_name}/bin/activate
   ```
   On Windows (CMD):
   ```bash
   {venv_name}\Scripts\activate
   ```

   Make sure the virtual environment is activated by checking that your terminal shows the name of the virtual environment before the prompt.

1. **Install Required Dependencies**  
   After cloning the repository in the previous step, navigate into the `Thesis` directory and install the required dependencies using the `requirements.txt` file:
   
   ```bash
   cd Thesis
   pip3 install -r requirements.txt
   ```

   The `requirements.txt` file includes the following dependencies:
   ```text
   numpy
   pandas
   tensorflow==2.12.0
   matplotlib
   jupyter
   seaborn
   scikit-learn
   streamlit
   ```

   **Note:** Ensure all dependencies are installed correctly before moving forward by running:
   ```bash
   pip3 freeze
   ```
   
1. **Extract the Thesis Repository**  
   Ensure that the downloaded `Thesis` folder is placed inside your virtual environment directory. You can navigate to the virtual environment folder and place the repository there, so the project files are available.

---

### Usage

1. **Open Jupyter Notebook**  
   Navigate to the `Thesis` directory and open Jupyter Notebook:
   ```bash
   cd Thesis
   python3.10 -m notebook
   ```
   **For Windows users**, the same command applies as long as Jupyter is installed in the environment.

   This will open the Jupyter interface in your browser. From there, open the `Tomato_Leaf_Disease_Detection_Classification.ipynb` notebook file to interact with the code and run experiments or validate the results.

1. **Stopping Jupyter Notebook**  
   After working with Jupyter Notebook, close it by pressing `Ctrl+C` in the terminal.
   
1. **Run the Web Application**  
   To run the web application for disease detection:
   - Navigate to the `Website` directory:
     ```bash
     cd Website
     ```
   - Run the Streamlit application:
     ```bash
     streamlit run predictions.py
     ```

   The application will launch in your web browser, where you can upload tomato leaf images and receive diagnoses.

1. **Stopping Streamlit Application**  
   After finishing the interaction with the website, close it by pressing `Ctrl+C` in the terminal.

1. **Deactivate the Virtual Environment**  
   Once you're finished working with the project, deactivate the virtual environment by running:
   ```bash
   deactivate
   ```

---

### Project Structure

The repository is organized as follows:
```
Thesis/
│
├── Dataset_Creation/               # Initial dataset used to create the new dataset
├── Models/                         # Saved models after being trained
├── Model_Evaluation_Results/       # Evaluation results of the trained models
├── Models_Training_History/        # Training history of models for each dataset split
├── My_Custom_Images/               # Personal test images used for testing purposes
├── Tomato_Leaves_Dataset/          # The dataset used for analysis in this thesis
├── Training_Times/                 # JSON files containing training times for each model and split
├── Website/                        # Web app built with Streamlit for user interaction
│   └── predictions.py              # Main file for running the web application
├── Tomato_Leaf_Disease_Detection_Classification.ipynb  # Notebook containing thesis results and interaction
└── requirements.txt                # File with required dependencies for easy installation
```

---

### Contributing

Contributions are welcome! If you encounter any issues or have suggestions for improvements, feel free to open a pull request or an issue in the repository.

---

### License

This project is licensed under the MIT License. See the [LICENSE](LICENSE) file for more information.

---

### Thesis Documentation

For a more detailed understanding of the project, including GPU setup for model training, it is recommended to start by reading the thesis PDF, which is available in the repository.

---

### Note on Notebook File Size

**Important:** The Jupyter notebook uploaded to this repository exceeds GitHub's rendering limit (15MB), which prevents it from being displayed directly on the platform.

#### Workarounds:
- **To view the notebook locally:**
  1. Clone the repository using:
     ```
     git clone <repository-url>
     ```
  1. Open the notebook locally using Jupyter or any compatible notebook viewer (e.g Visual Studio Code).
  
- **Alternatively, view the notebook on another platform:**
  - You can use [nbviewer](https://nbviewer.jupyter.org/) or upload the notebook to [Google Colab](https://colab.research.google.com/) for better viewing.
