import dataset
from pathlib import Path
import numpy as np
import torch
from CNN import CNN

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

# df_test.to_pickle(Path("annotations/test_labels.pkl"))
# df_val.to_pickle(Path("annotations/val_labels.pkl"))
# df_train.to_pickle(Path("annotations/train_labels.pkl"))

# exploration
for i in range(len(list(df_test['label']))):
    #if df_test['label'].iloc[i][30] == 1 and df_test['label'].iloc[i][31] == 1:
    if df_test['label'].iloc[i][37] == 1 and df_test['label'].iloc[i][22] == 1:
        print(df_test.iloc[i]['file_path'])
        print(df_test.iloc[i]['label'])
    if df_test['label'].iloc[i][19] == 1 and df_test['label'].iloc[i][22] == 1:
        print(df_test.iloc[i]['file_path'])
        print(df_test.iloc[i]['label'])
    if df_test['label'].iloc[i][19] == 1 and df_test['label'].iloc[i][37] == 1:
        print(df_test.iloc[i]['file_path'])
        print(df_test.iloc[i]['label'])
# files = [
#     "val/d/katherine_roberts_perl-j_s__bach_french_suites-16-suite_no_3_in_b_minor_bwv_814_sarabande-204-233.npy",
#     "val/d/paul_avgerinos-maya__the_great_katun-03-dawn_of_the_gods_part_3-407-436.npy",
#     "val/d/paul_avgerinos-maya__the_great_katun-04-dawn_of_the_gods_part_4-88-117.npy",
#     "val/e/solace-balance-10-tiger_moon_dance-0-29.npy",
#     "val/e/touchinggrace-submission-08-eastern_sky-523-552.npy",
#     "val/e/yongen-moonrise-07-vertigo-175-204.npy",
#     "val/f/dac_crowell-redshifted_harmonies-01-tranquilitatis-1074-1103.npy",
#     "val/f/dac_crowell-redshifted_harmonies-01-tranquilitatis-494-523.npy",
#     "val/f/dac_crowell-redshifted_harmonies-02-prinsengracht_extensions-1306-1335.npy",
#     "val/f/magnatune-relaxation_spa-04-secret_solution_michael_masley-146-175.npy",
#     "val/f/magnatune-relaxation_spa-12-northeastern_corridor_dac_crowell-291-320.npy",
#     "val/f/magnatune-relaxation_spa-12-northeastern_corridor_dac_crowell-726-755.npy"
# ]

# path_annotations_test = Path("annotations/test_labels.pkl")
# test_dataset = dataset.MagnaTagATune(path_annotations_test, Path("samples/"))
# test_loader = torch.utils.data.DataLoader(
#     test_dataset,
#     shuffle=False,
#     pin_memory=True,
# )
# best_state = torch.load(Path("models/CNN_batch_size=10_epochs=30_learning_rate=0.0075_momentum=0.95_dropout=0_length_conv=256_stride_conv=256_normalisation=minmax_outchannel_stride=32_model=Basic_inner_norm=None_run_27"))
# best_model = CNN(**best_state["kwargs"])
# best_model.load_state_dict(best_state["model"])
# best_model = best_model.to(torch.device("cuda"))
# best_model.eval()
# with torch.no_grad():
#     for filename, batch, labels in test_loader:
#         if 'val/d/beth_quist-shall_we_dance-02-monarch_dance-204-233.npy' in filename:
#             logits = best_model(batch.to(torch.device("cuda")))
#             print(torch.where(logits>0.1, 1,0))
#             print(labels)
