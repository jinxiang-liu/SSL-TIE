python -u main.py \
--batch_size 1 --gpus 2 \
--exp_name SSL-TIE \
--dataset_mode VGGSound \
--test_set Flickr \
--soundnet_test_path "dir_of_SoundNet_Test_Data/" \
--test "ckpts/vggsound144k.pth.tar"