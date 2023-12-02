import dataset
from pathlib import Path
import numpy as np

# path_annotations = Path("annotations/original_train_labels.pkl")
# path_samples = Path("samples/original_train")
# audio_data = dataset.MagnaTagATune(path_annotations, path_samples)
# audio_data_test = dataset.MagnaTagATune(Path("annotations/original_test_labels.pkl"), Path("samples/test"))

# # df_oldtrain = audio_data.dataset
# #df_test = audio_data_test.dataset

# # df_oldtrain["file_path"] = df_oldtrain["file_path"].str.replace('train/', '')
# # df_val = df_oldtrain.groupby("part").get_group("c")
# # df_train = df_oldtrain[df_oldtrain['part'] != 'c']

# #df_test["file_path"] = df_test["file_path"].str.replace('val/', '')

# # print(df_test)
# # print(df_val)
# # print(df_train)
# # df_test.to_pickle(Path("annotations/test_labels.pkl"))
# # df_val.to_pickle(Path("annotations/val_labels.pkl"))
# # df_train.to_pickle(Path("annotations/train_labels.pkl"))



path_annotations = Path("annotations/original_train_labels.pkl")
path_samples = Path("samples/train")
audio_data_train = dataset.MagnaTagATune(path_annotations, path_samples)
audio_data_test = dataset.MagnaTagATune(Path("annotations/original_val_labels.pkl"), Path("samples/test"))

df_oldtrain = audio_data_train.dataset
df_test = audio_data_test.dataset
df_val = df_oldtrain.groupby("part").get_group("c")
df_train = df_oldtrain[df_oldtrain['part'] != 'c']

df_test.to_pickle(Path("annotations/test_labels.pkl"))
df_val.to_pickle(Path("annotations/val_labels.pkl"))
df_train.to_pickle(Path("annotations/train_labels.pkl"))

