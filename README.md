# Snakes Challenge
Based on the [AI Crowd Snake Species Identification Challenge](https://aicrowd.com/challenges/snake-species-identification-challenge)

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

# Notes 
* We can use any method we like
* Stats/NN/CNN/etc
* Any combination of methods
* Get the dataset
* Apply techniques
* Understand what is and isn't working
* Explain what we did in the report
* Talk about what method we used
* The accuracy and more
* Some interesting metrics to use:
  * Precision
  * Recall 
    * There is a graph that can encapsulate these two fairly well
  * F1 score
  * Cohen's $\kappa$ (number between -1 and 1, that is somewhat similar to correlation), what accuracy we expect to get from chance
    * Anything above > 0.6 is doing well (humans normally get > 0.6)
    * Cohen's kappa takes the data's bias into account
  * Confusion Matrix
* Doesn't expect us to solve it explicitly
* Two people in a group, every group to submit their result to the competition
* Try different architectures
* Different hyperparameters
* Analyse results carefully
* Retraining
* Auto-encoders
* etc
* The more stuff we can try the more he will be impressed