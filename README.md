# TypeWriter
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

+ ...  

### RUN!!!:
+ Change the paths and configurations in your config file. 

+ Run the model:  
```python3 main.py -config [path_to_config_file]```

  - You can monitor the model's loss, accuracy, and f1-score using tensorboard:  
  ```tensorboard --logdir [path_to_tensorboard_dir]```

