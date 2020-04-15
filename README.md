# Learnable Visual Rhythms Based on the Stacking of Convolutional Neural Networks for Action Recognition

## 1.vr_extraction: Extract visual rhythm (first CNN in the stack)

```
python3 get_visual_rhythm.py -d <dataset> -f <dataset_path> -a <architecture> -l <list>
```

* \<dataset\>: ucf101,hmdb51
* \<dataset_path\>: path to the RGB frames
* \<architecture\>: inception_v3,resnet152
* \<list\>: dataset_list_ucf.txt, dataset_list_hmdb.txt
* Other parameters:
  * -e \<batch\>: External batch, control the RAM usage
  * -i \<batch\>: Internal batch, control the GPU memory usage
  * -s \<index\>: Start instance
  * -t \<index\>: End instance
  * -c \<checkpoint_path\>: Checkpoint file. If none is indicated, it uses the ImageNet weights
  
 The rhythms are saved in \<architecture\>_\<dataset\>_VR\<i\>, where i corresponds to the depth.

 
## 2.vr_normalization: Match the second CNN input dimension

```
python3 preprocessing.py -src_dir <architecture>_<dataset>_VR<i> -new_width <width> -wm <method> -new_height <height> -hm <method> -ouput_dir <output_path> -ext png
```

* -wm: horizontal method: none (keep original width), sim_ext (symmetric extension), resize
* -hm: vertical method: none, pool

## 3.vr_stream: Classify the visual rhythms (second CNN)

### Training 

```
python3 main_single_gpu.py <input_path> -d <dataset> -a <architecture> -s <split> -b 4 --new_width 0 --new_height 0 --n_images 10 --epochs 100 --iter-size 10
```

### Test

```
python3 rhythm.py -d <dataset> -s <split> -a <arquitecture> -b <batch> <input_path> <checkpoint>
```




