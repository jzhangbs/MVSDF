# Prepare your own data
This is an instruction about how to process your own data. We take the BlendedMVS dataset as an example. You may need to change some parameters accordingly (image resolution, probability threshold, etc.). 

## VisMVSNet
Run [MVSNet](https://github.com/jzhangbs/Vis-MVSNet). Suppose you have the camera files and the view selection file. Please arrange the input files as follows.
```
- <data_root>/<scan>
    - cams
    - images
    - pair.txt
```
You can also refer to "[Quick test on your own data](https://github.com/jzhangbs/Vis-MVSNet#quick-test-on-your-own-data)" section. 

Then run VisMVSNet. 
``` sh
DATA_ROOT=
SCAN=
OUT_DIR=
NUM_SRC=2

mkdir -p ${OUT_DIR}
python test.py \
    --data_root ${DATA_ROOT}/${SCAN} \
    --dataset_name general \
    --model_name model_cas \
    --num_src ${NUM_SRC} \
    --max_d 256 \
    --interval_scale 1 \
    --resize 768,576 \
    --crop 768,576 \
    --write_result \
    --result_dir ${OUT_DIR}
```

## Point cloud fusion
Run [point cloud fusion](https://github.com/jzhangbs/pcd-fusion). 
``` sh
python fusion.py \
    --data ${OUT_DIR} \
    --pair ${OUT_DIR}/pair.txt \
    --view 10 \
    --vthresh 2 \
    --pthresh .8,.7,.8 \
    --cam_scale 1 \
    --no_normal \
    --downsample -1
```

## Cut the point cloud
Copy the `all_torch.ply` to `cut.ply` and cut it. This cut point cloud is used to determine the bounding box. 

## Convert data format
Now we convert the VisMVSNet format to MVSDF format. 
``` sh
mkdir -p ${OUT_DIR}/imfunc4/image_hd
mkdir -p ${OUT_DIR}/imfunc4/mask_hd
mkdir -p ${OUT_DIR}/imfunc4/depth
python datasets/vismvsnet2mvsdf.py \
    --data_root ${OUT_DIR} \
    --range_source pcd \
    --pthresh .8,.7,.8 \
    --prob_mask \
    --resize 768,576 \
    --crop 768,576 \
    --ext_image_path ${DATA_ROOT}/${SCAN}/images/{:08}.jpg
    # --ext_image_from_one  # set this if the image index starts from 1
```