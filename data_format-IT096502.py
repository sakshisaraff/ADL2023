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

# # print(np.load(Path("../samples/train/0/williamson-a_few_things_to_hear_before_we_all_blow_up-12-a_please_goodbye_from_whore-59-88.npy")))


path_annotations = Path("annotations/original_train_labels.pkl")
path_samples = Path("samples/test")
audio_data = dataset.MagnaTagATune(path_annotations, path_samples)
audio_data_test = dataset.MagnaTagATune(Path("annotations/original_test_labels.pkl"), Path("samples/val"))

# df_oldtrain = audio_data.dataset
#df_test = audio_data_test.dataset

# df_oldtrain["file_path"] = df_oldtrain["file_path"].str.replace('train/', '')
# df_val = df_oldtrain.groupby("part").get_group("c")
# df_train = df_oldtrain[df_oldtrain['part'] != 'c']

#df_test["file_path"] = df_test["file_path"].str.replace('val/', '')

# print(df_test)
# print(df_val)
# print(df_train)
# df_test.to_pickle(Path("annotations/test_labels.pkl"))
# df_val.to_pickle(Path("annotations/val_labels.pkl"))
# df_train.to_pickle(Path("annotations/train_labels.pkl"))

# print(np.load(Path("../samples/train/0/williamson-a_few_things_to_hear_before_we_all_blow_up-12-a_please_goodbye_from_whore-59-88.npy")))