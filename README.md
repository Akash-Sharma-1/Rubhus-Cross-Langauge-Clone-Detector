# Rubhus-Cross-Langauge-Clone-Detector

This repository contains the source code for the paper - "Improving Cross-Language Code CloneDetection via Code Representation Learning and Graph Neural Networks"


## Code Organisation 📜
Current organisation contains files pertaining to models (`rubhusModel.py, baselineModel.py`), trainers (`trainerBaseline.py , trainerRubhus.py`) and some helper function file.  

    Repository
    ├── helper functions
    ├── models
    └── trainers
   
After setting up the repository, it would contain dataset files as well.

## Setting Up ⚙

### Clone the repo

       git clone https://github.com/Akash-Sharma-1/Rubhus-Cross-Langauge-Clone-Detector.git

### Installing Dependencies

       pip install -r requirements.txt

Note - Pytorch and Pytorch-Geometric (+ associated dependencies) versions must be installed in accordance the compatablity of Cuda version and operation system 

### Datasets
#### Extraction of Dataset Files
- Java-Python Dataset - [Link](https://drive.google.com/file/d/1pOkkNpc9lmMXME8mCUYJRjl_-5GJzB6f/view?usp=sharing)  
- C-Java Dataset - [Link](https://drive.google.com/file/d/1pOkkNpc9lmMXME8mCUYJRjl_-5GJzB6f/view?usp=sharing)

#### Setting up Dataset Files
- Unzip the downloaded files and extract the datasets files.
- Place these extracted files in the root directory of this repository

### Configuration of file paths
- .

## Usage 💫

### Training RUBHUS Model
       python3 trainerRubhus.py

### Training Baseline Model
       python3 trainerBaseline.py
      
### Results 

## About the original setup
- In our experiments we have trained Rubhus and Baseline Models for x and y epochs for Java Python Dataset and x2 and y2 epochs for C-Java Dataset. 
- The hyperparameters used in the original experiments as well as in this source code are reported in the paper.
- We have used GTx 2080Ti GPU to run our experiments. The time analysis of the tool also has been reported in the paper.

## Citing the project 📑

If you are using this for academic work, we would be thankful if you could cite the following paper.
`BIBTEX`

```
@{,
 author = {},
 title = {Improving Cross-Language Code CloneDetection via Code Representation Learning and Graph Neural Networks},
 ....
}
```
