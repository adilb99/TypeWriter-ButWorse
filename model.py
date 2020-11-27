import pickle

def main():
	 with open("arg_data_training.pkl", "rb") as f:
        arg_data_training = pickle.load(f)

 	with open("arg_data_validation.pkl", "rb") as f:
        arg_data_validation = pickle.load(f)

    with open("ret_data_training.pkl", "rb") as f:
        ret_data_training = pickle.load(f)

    with open("ret_data_validation.pkl", "rb") as f:
        ret_data_validation = pickle.load(f)

if __name__ == "__main__":
    main()