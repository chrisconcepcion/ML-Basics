# ML-Basics Directory Structure
ml_basics/
│
├── data/                  # For storing datasets
├── notebooks/            # For Jupyter notebooks
└── scripts/              # For Python scripts
    ├── __init__.py
    ├── linear_regression.py
    ├── logistic_regression.py
    ├── decision_trees.py
    └── knn.py


## Install venv if not already installed
sudo apt install python3-venv

# Create virtual environment
python3 -m venv venv

# Activate virtual environment
source venv/bin/activate

Install required packages

# Upgrade pip
pip install --upgrade pip

# Install required packages
pip install numpy pandas scikit-learn matplotlib seaborn jupyter

# Install Jupyter if not installed via previous step
pip install jupyter

# Start Jupyter Notebook
jupyter notebook




# Create virtual environment
python3 -m venv venv

# Activate the virtual environment
source venv/bin/activate

# Run the script
python3 scripts/basic_ml_workflow.py