python -u main.py \
--batch_size 1 --gpus 1 \
--exp_name SSL-TIE \
--dataset_mode VGGSound \
--test_set VGGSS \
--vggss_test_path "dir_of_vggsound_testset" \
--test "ckpts/vggsound144k.pth.tar"