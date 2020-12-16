import csv
import numpy as np

def load_csv_data(data_path):
    all_data = np.genfromtxt(data_path, delimiter=",", skip_header=1)
    ids = all_data[:, 0]
    v = all_data[:, 1]
    x = all_data[:, 2:]
    
    return ids, x, v

def create_csv_dataset(ids, x, v, name):
    with open(name, 'w') as csvfile:
        fieldnames = ['city id', 'normalized population', 'position x', 'position y']
        writer = csv.DictWriter(csvfile, delimiter=",", fieldnames=fieldnames)
        writer.writeheader()
        
        for i in range(len(ids)):
            writer.writerow({'city id':ids[i],'normalized population':v[i],'position x':x[i][0],'position y':x[i][1]})
            
def create_csv_submission(ids, selection, name):
    with open(name, 'w') as csvfile:
        fieldnames = ['city id', '1/0 variable']
        writer = csv.DictWriter(csvfile, delimiter=",", fieldnames=fieldnames)
        writer.writeheader()
        
        for i in range(len(ids)):
            
            writer.writerow({'city id':ids[i],'1/0 variable':int(selection[i])})