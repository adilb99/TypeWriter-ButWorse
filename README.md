# TypeWriter

  ! Short project summary here !

-------------------  
## Running guide:  

### Prerequisites:  

+ It is recommended to use virtual environment
  -  Create and activate virtual environment:  
```virtualenv -p python3 venv  ```  
```source venv/bin/activate ```

+ To install dependencies, run:  
```pip3 install -r requirements.txt```

### Data Extraction and Preparation:  

+ Extract the training.tar.gz and validation.tar.gz archives of raw .py files in the ./data directory 

+ Open extractor.py and uncomment the commented code in the ``` main ``` function to extract and vectorize data from .py files

+ Alternatively, keep the code commented and run it as is to use/open pickles of already extracted code. (in the ./pickles directory)

+ Adjust the directory path in the code to store .npy files after processing.

+ The .npy files are then ready to be used by the model

### Running the model:
+ Change the paths and configurations in your config file. 

+ Run the model:  
```python3 main.py -config [path_to_config_file]```

  - You can monitor the model's loss, accuracy, and f1-score using tensorboard:  
  ```tensorboard --logdir [path_to_tensorboard_dir]```

