# Disaster Response Pipeline Webapp

## Overview

This repo is an extension of [this one](https://github.com/russom/DSND-P3-Disaster-Response-Pipeline), containin code for the creation of a Disaster Response webapp capable of classifying messages from user based on a ML model trained for that purpose.

The code in the original repo contains all the scripts needed to handle the data and train the classifier: it also includes the cose for a webapp capable of running locally on your machine.
The code in this repo focuses only on the webapp part, with the scope of packaging it for a cloaud deployment. The target platform chosen is [Heroku](https://www.heroku.com/).
The code has also been refactored with respect to what present in the original repo.


## Requirements
In order to run the app locally I have prepared an [`environment.yml`](./environment.yml) file to be used to install an environment with [Anaconda](https://www.continuum.io/downloads):

```sh
conda env create -f environment.yml
```

After the installation the environment should be visible via `conda info --envs`:

```sh
# conda environments:
#
dsnd-proj3        /usr/local/anaconda3/envs/dsnd-proj3
...

```

the full list of requirements that get loaded is also visible in [`requirements.txt`](./requirements.txt), that will be used by Heroku.

## Instructions
Before running th code you will need to download the model file availble [here](https://drive.google.com/file/d/13A-E9P84fXXDcGOghdw1zF3jxirvAuak/view?usp=sharing), and move it to the [`data`](./data) folder. Please note that:

* In order to allow the upload to Heroku, the original pickle file had to be compressed using the instructions found [here](https://betterprogramming.pub/load-fast-load-big-with-compressed-pickles-5f311584507e). This led to some modifications in the code and a bit more time to load the app.
* The file is nonetheless too big to be uplaoded to git directly, so I put it in my google drive.
* A database with the cleaned results is already available in the same folder

After downloading the model, if you want to run the app locally, after having activating the conda env you will need to uncomment the last line of [`app.py`](./app.py):

```
# uncomment to run locally
# app.run(host='0.0.0.0', port=3001, debug=False)
```

And just run `python app.py`.

## License
 <a rel="license" href="http://creativecommons.org/licenses/by-nc-nd/4.0/"><img alt="Creative Commons License" style="border-width:0" src="https://i.creativecommons.org/l/by-nc-nd/4.0/88x31.png" /></a><br />This work is licensed under a <a rel="license" href="http://creativecommons.org/licenses/by-nc-nd/4.0/">Creative Commons Attribution-NonCommercial-NoDerivatives 4.0 International License</a>.
