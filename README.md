# PyTorch Intro, Data Analysis & Neural Network Models

Introductory deep learning assignment covering PyTorch fundamentals, data analysis,
classical ML models, and neural network implementation from scratch.  
**Course:** CSE 676-B — Deep Learning, University at Buffalo (Spring 2025)

## Overview
This project covers four core components:
- **Part I:** PyTorch fundamentals — tensors, autograd, model building, TensorBoard
- **Part II:** Real-world data analysis, ML model comparison, and shallow NN
- **Part III:** OCTMNIST retinal disease classification using CNN
- **Part IV:** Theoretical derivations — forward/backward pass and tanh derivative proof

## Tech Stack
- **Language:** Python
- **Framework:** PyTorch
- **Libraries:** NumPy, Pandas, Matplotlib, Seaborn, Scikit-learn,
  medmnist, torchinfo, TensorBoard
- **Tools:** Jupyter Notebook

## Project Structure
├── a0_part_1.ipynb      # PyTorch tutorial walkthrough
├── a0_part_2.ipynb      # Data analysis, ML models, shallow NN
├── a0_part_3.ipynb      # OCTMNIST CNN classification
├── a0_report.pdf        # Theoretical derivations (forward/backward pass + tanh)
└── a0_weights.txt       # Link to saved model weights

## Dataset
**Part II:** Real-world dataset (20K+ entries) sourced from Open Data Buffalo /
US Government Data / Yahoo Finance  
**Part III:** [OCTMNIST](https://medmnist.com/) — 109,309 optical coherence tomography
(OCT) images for retinal disease classification across 4 classes (28x28 grayscale)

## Key Results
| Part | Task | Target |
|------|------|--------|
| II | ML model comparison (3 algorithms) | >65% accuracy |
| II | Shallow NN | >75% accuracy |
| III | Base CNN (OCTMNIST) | >75% accuracy |
| III | Improved CNN (OCTMNIST) | >80% accuracy |

## Key Concepts Covered
- PyTorch tensors, autograd, and training loops
- Data preprocessing — normalization, missing value handling, one-hot encoding
- ML models: Logistic Regression, Decision Tree, Random Forest (scikit-learn)
- CNN architecture design with convolutional and fully connected layers
- Regularization: dropout, early stopping, batch normalization, learning rate scheduling
- Evaluation: confusion matrix, ROC curve, precision, recall, F1, TensorBoard
- Theoretical: forward/backward pass derivation, tanh derivative proof

## How to Run
```bash
git clone https://github.com/rishtha/PyTorch-Intro-Data-analysis-NN-Models.git
cd PyTorch-Intro-Data-analysis-NN-Models
pip install -r requirements.txt
pip install medmnist
jupyter notebook
```

## Author
Rishitha Saravanan Priya  
[LinkedIn](https://linkedin.com/in/rishithasp) | [Portfolio](https://rishitha.dev)
