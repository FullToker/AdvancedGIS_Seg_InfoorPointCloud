# Advanced GIS Project -- Segementing Indoor Point Clouds


## mmdetection3d
### Docker Environment

1. Edit the original dockerfile
```Dockerfile
&& pip install --no-cache-dir -r ./requirements.txt
```

2. Run the dockerfile under the mmdetection3d
```zsh
docker build -t mmdetection3d docker/ 
```

3. Build the container
```zsh
docker run -it --name container1 --gpus all \
    -v {/path/to/your/directory}:/workspace \
    -w /workspace \
     mmdetection3d /bin/bash
mim install "mmdet3d>=1.1.0rc0"
docker stop conatiner1
```
If has done this before, then run:
```zsh
docker start container1
docker exec -it container1 /bin/bash
```
or 
```
docker start -ai container1
```

Don't forget to stop the container in the end.

4. Interference from pre-trained model

Remeber to check the configs and checkpoint files.
```zsh
python demo/pcd_seg_demo.py demo/data/scannet/scene0000_00.bin pointnet2_msg_2xb16-cosine-250e_scannet-seg-xyz-only.py ${CHECKPOINT_FILE} [--out-dir ${OUT_DIR}] [--show]
```
Check the output in workplace/output. Normaly, the checkpoint files are saved in demo/cps.