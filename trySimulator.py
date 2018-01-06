import pandas as pd

from plugins.csv_collect_and_publish import CollectAndPublish

project_path = "/home/geesara/project/OpenBCI_Python"
dataset_location = project_path+ "/build/dataset/inittryout.csv"

publisher = CollectAndPublish(project_path)
raw_data = pd.read_csv(dataset_location).ix[:, 0:4].dropna()
publisher.main_thread.start()
for i in range(0, len(raw_data.ix[:, 0])):
    sample={}
    sample['id']=i%2000
    sample['channel_data'] = raw_data.ix[i].values
    publisher.getting_next_point(sample)

