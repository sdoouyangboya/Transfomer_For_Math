## Project Overview

### Data Structure
- **Data Directory (`./data`)**: Houses `train.txt`, `validation.txt`, and `test.txt`, which form the complete dataset.
- **Vocabulary Directory (`./vocab`)**: Stores the vocabularies for both source and target sequences.
- **Model Directory (`./model`)**: Contains the `best_model.pt`, which is the trained model file.
- **Output Directory (`./output`)**: Includes `predictions.txt`, detailing the model's predictions.

### Source Code
- **Model Files**: `backbone.py` and `transformer.py` define the transformer model's architecture.
- **Data Management**: `data.py` handles the random splitting of the dataset into train, validation, and test sets (pre-split dataset included).
- **Training Script**: `train.py` manages the model training with adjustable configurations.
- **Testing Script**: `test.py` conducts predictions and accuracy computations using the trained model on the test dataset.
- **Dependencies**: `requirements.txt` enumerates all necessary packages.
- **Model Summary**: `network.txt` provides an overview of the model and its parameters.

### Setup and Installation
1. **Environment Creation**:
    ```shell
    conda create --name <env_name> python=3.9.2
    ```
2. **Environment Activation**:
    ```shell
    conda activate <env_name>
    ```
3. **Dependency Installation** (run in `/Attention` directory):
    ```shell
    pip install -r requirements.txt
    ```

## Execution Instructions

### Data Setup
- **Dataset Splitting**:
    ```shell
    python data.py
    ```

### Model Training
- **Train Execution** (configuration adjustments can be made in `train.py` or via CLI):
    ```shell
    python train.py
    ```

### Model Evaluation
- **Test Evaluation**:
    ```shell
    python test.py
    ```
### Blind Test Evaluation
- Execute performance evaluation on a blind `test.txt` with `main.py`:
    ```shell
    conda create -y -n homeworkenv
    conda activate homeworkenv
    cd $HOMEWORK_CODE_DIRECTORY
    pip install -r requirements.txt
    python main.py  
    ``` 

## Model Metrics
The model demonstrates an accuracy of `81.887%` based on **strict equality** comparison between predicted and true target sequences. This metric reflects the outcome after 20 epochs on a single GPU.

Refer to `Report.pdf` in the repository for an in-depth exploration of the methodology, parameter selection, and performance analysis.