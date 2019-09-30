# CreditCardML2.0
Credit card fraud detector using multiple algorithms to find best fit.

## Requirements
* Python 3.7
* pip3

### Installs
When python 3.7 and pip3 has been installed you need to install the requirements either to your
virtual environment or global using the requirements file.
```bash
pip3 install -r requirements.txt
```

## Usage

### Short version
The short version is only to test the functionality. Since the dataset is pretty large, the code and its functionality can
be tested with only training on the first 100 instances in the dataset after sub-sampling and pre processing. This does
not give you good results but perhaps you dont have alot of time to validate the behaviour and thus could review the report
made on the experiment or look up all charts made from the experiment.

#### Instructions
1. Make sure all requirements above are met.
2. Open the main.py file and make sure in the CONSTANTS section the following values are set:
    * DATASET_SIZE = 100
    * DEBUG = False
    * LOCAL = True
    * GCP = False
    * VERSION = '1'
3. Run the python command for the main.py file from root directory.
    ```bash
    python3 main.py
    ```
4. Get a cup of coffee.
5. When the training finishes the results are all printed to console
    * The trained model files will be stored under ./data/models
    * All results including graphs and a special result file are in ./data/models

### Full Version
The full version can be made on a GCP instance to speed up the process however you will need to either
configure new buckets for storage and training instance and then download a credential file for those.
However you can go ahead and train the entire dataset on all algorithms in local environment ... but it
will take time ... A LOOONG TIME!

#### Instructions
1. Make sure all requirements above are met.
2. Open the main.py file and make sure in the CONSTANTS section the following values are set:
    * DATASET_SIZE = 100
    * DEBUG = False
    * LOCAL = False
    * GCP = False
    * VERSION = '1'
3. Run the python command for the main.py file from root directory.
    ```bash
    python3 main.py
    ```
4. Get a cup of coffee ... Then some more!
5. When the training finishes the results are all printed to console
    * The trained model files will be stored under ./data/models
    * All results including graphs and a special result file are in ./data/models

## Reports

Saved models, results and graphs can be found in ./data/pre_trained folder.