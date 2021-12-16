# Computer vision

## Installation
####**Create a conda environment named `computer_vision` and activate it:**

`conda create --name computer_vision`

`conda activate computer_vision` 

#### **Install the dependencies**

`pip install -r requirements.txt`


## Create dataset
1. `mkdir dataset`
2. `python utils/get_annotations_bis.py`
3. `python -c 'from utils.dataset import generate_dataset; generate_dataset()`

