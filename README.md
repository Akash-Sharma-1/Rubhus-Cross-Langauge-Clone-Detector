This repository contains the source code for the paper - "Improving Cross-Language Code CloneDetection via Code Representation Learning and Graph Neural Networks"


## Code Organisation
Current organisation contains files pertaining to models (`rubhusModel.py, baselineModel.py`), trainers (`trainerBaseline.py , trainerRubhus.py`) and some helper functions file.  

    Repository
    ├── helper functions
    ├── models
    └── trainers
   
After setting up the repository, it would contain dataset files as well.

## Setting Up

### Clone the repo

       git clone https://github.com/Akash-Sharma-1/Rubhus-Cross-Langauge-Clone-Detector.git

### Installing Dependencies

       pip install -r requirements.txt

Note - Pytorch and Pytorch-Geometric (+ associated dependencies) versions must be installed in accordance the compatablity of Cuda version and operation system 

### Datasets
#### Extraction of Dataset Files
Java-Python Dataset - Link  
C-Java Dataset - Link
#### Setting up Dataset Files
- Unzip the downloaded files and extract the datasets files.
- Place these files in the root directory of this repository


### Configuration of file paths


## Usage

### Training RUBHUS Model
       python3 trainerRubhus.py

### Training Baseline Model
       python3 trainerBaseline.py
      
### Results 


## Citing the project

If you are using this for academic work, we would be thankful if you could cite the following paper.
`BIBTEX`

```
@{,
 author = {},
 title = {Improving Cross-Language Code CloneDetection via Code Representation Learning and Graph Neural Networks},
 ....
}
```