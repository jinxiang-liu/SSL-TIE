python -u main.py \
--batch_size 1 --gpus 1 \
--exp_name SSL-TIE \
--seed 4 \
--dataset_mode Flickr \
--soundnet_test_path "dir_of_SoundNet_Test_Data/" \
--test "ckpts/flickr144k.pth.tar"