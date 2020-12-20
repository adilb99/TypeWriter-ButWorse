# TypeWriter, But Worse

## Summary

  This project is an attempt at neural type prediction in Python code inspired by the paper "TypeWriter: Neural Type Prediction with Search-based Validation" by Michael Pradel, Georgios Gousios, Jason Liu, Satish Chandra.

  This variant features a slightly different approach to feature extraction and overall ML pipeline. 

  Here's how our pipeline looks:

  ![pipeline picture](https://i.ibb.co/ryDCwFk/pipeline.png)

-------------------  
## Running guide:  

### Prerequisites:  

+ It is recommended to use virtual environment
  -  Create and activate virtual environment:  
```virtualenv -p python3 venv  ```  
```source venv/bin/activate ```

+ To succesfully run the extractor, Python 3.9 is needed as this project uses functions that are only available with this October 2020 Python release. 

+ To install dependencies, run:  
```pip3 install -r requirements.txt```

### Data Extraction and Preparation:  

+ Extract the training.tar.gz and validation.tar.gz archives of raw .py files in the ./data directory 

+ Currently, ``` main ``` function in the extractor.py uses pre-processed pickles of data in the ./pickles directory. Run extractor.py to use those pickles and convert them to .npy files that can be used by the model.

+ Alternatively, comment/uncomment parts of the code in the ``` main ``` function as specified inside of it to run extraction on the raw .py files extracted earlier.

+ Then, it should create necessary .npy files in the newly made ./numpy directory

+ The .npy files are then ready to be used by the model.

### Running the model:
+ Change the paths and configurations in the config.json file. 

+ Run the model:  
```python3 main.py -config [path_to_config_file]```

  - You can monitor the model's loss, accuracy, and f1-score using tensorboard:  
  ```tensorboard --logdir [path_to_tensorboard_dir]```

