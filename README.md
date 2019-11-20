# Snakes Challenge
![Banner](banner.png)

Based on the [AI Crowd Snake Species Identification Challenge](https://aicrowd.com/challenges/snake-species-identification-challenge)

# Task
The _Snake Species Identification Challenge_ or _SSIC_ is a dataset containing 82601 images of snakes with varying backgrounds, crops and viewpoints. The goal of SSIC is to classify each snake contained in each image into one of 45 classes or snake species.

The goal of this project is to investigate the application of various computer vision approaches to this image classification task for the SSIC challenge. Additionally the project investigates why these methods perform differently.

# Cloning This Repo

Make sure to include submodules
```
git clone --recursive <repo_url>.git
```

Or if you have already cloned the repo, but are missing submodules
```
git submodule update --init
```

# Setup
Run the following command to create a new conda env with the required packages:
```
conda env create -f environment.yml
```
To run jupyter notebooks you will need to add another kernel, do this as follows:
```
conda activate snakes
python -m ipykernel install --user --name=snakes
```

If you install more packages and need to add them to the `.yml` file, you can use this command:
```
conda env export > environment.yml
```

# Trained Models
We trained a number of models, they are available via [Google Drive](https://drive.google.com/drive/folders/1r1JVvd6E7hiZ2slUgI0UJ0JygGjYYj3M?usp=sharing)