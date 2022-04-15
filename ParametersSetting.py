import os


###############################################################
#base predict_root_path
input_root_path="D:\\Brats2018InputRoot"
output_root_path = "D:\\Brats2018OutputRoot"

########
raw_train_data_path=os.path.join(input_root_path, "RawDataTrain")
raw_test_data_path=os.path.join(input_root_path,"RawDataTest")
train_npy_path = os.path.join(input_root_path,"TrainNpy")
test_npy_path = os.path.join(input_root_path, "TestNpy")

if not os.path.isdir(test_npy_path):
    os.makedirs(test_npy_path)
if not os.path.isdir(raw_train_data_path):
    os.makedirs(raw_train_data_path)
if not os.path.isdir(output_root_path):
    os.makedirs(output_root_path)
if not os.path.isdir(train_npy_path):
    os.makedirs(train_npy_path)

training_result_folder= "result_ChannelAttention"
test_version_name= "BaseLine"
test_checkpoint_folder="result_SpaceAttention"

init_lr=5e-4
patch_size = (64, 64, 64)
mini_delta = 5e-4
batch_size = 1
val_batch_size = 1
step_size=0.5
num_class=4
max_epoch=300

vi_lastlayer_flag=True
hp = {"batch_size": batch_size,
      "patch_size": patch_size,
      "predict_root_path": output_root_path,
      "fold": 1,
      "val_batch_size": val_batch_size}
####################################################################

# raw_train_data_path = "/input/unzip_data"
# output_root_path= "/data/BrainTumorSegAttention/Brats2017ExperimentResults"
# train_npy_path= "/input/Brats2017TrainNpy"
#
# if not os.predict_root_path.isdir(output_root_path):
#     os.makedirs(output_root_path)
#
# init_lr=0.2
# patch_size = (128, 128, 128)
# mini_delta = 5e-4
# batch_size = 1
# val_batch_size = 2
# step_size=0.5
# num_class=4
# max_epoch=300
# training_result_folder= "result_ChannelAttention"
# test_version_name= "test_result_GridAttention"
# test_checkpoint_folder="result_SpaceAttention"
# vi_lastlayer_flag=True
# hp = {"batch_size": batch_size,
#       "patch_size": patch_size,
#       "predict_root_path": output_root_path,
#       "fold": 1,
#       "val_batch_size": val_batch_size}

##################################################
# raw_train_data_path = "D:\\Brats2018RawData"
# output_root_path = "D:\\Brats2017ExperimentResults"
# train_npy_path = "D:\\Brats2017TrainNpy"
# train_npy_path_all_mean="D:\\Brats2017TrainNpyAllMean"
#
# if not os.predict_root_path.isdir(output_root_path):
#     os.makedirs(output_root_path)
#
# init_lr=5e-4
# patch_size = (64, 64, 64)
# mini_delta = 5e-4
# batch_size = 1
# val_batch_size = 1
# step_size=0.5
# num_class=4
# max_epoch=300
# training_result_folder= "result_ChannelAttention"
# test_version_name= "test_result_GridAttention"
# test_checkpoint_folder="result_SpaceAttention"
# vi_lastlayer_flag=True
# hp = {"batch_size": batch_size,
#       "patch_size": patch_size,
#       "predict_root_path": output_root_path,
#       "fold": 1,
#       "val_batch_size": val_batch_size}



