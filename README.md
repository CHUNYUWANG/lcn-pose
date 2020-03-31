
# Quick start
Tensorflow Implementation of [Optimizing Network Structure for 3D Human Pose Estimation](http://openaccess.thecvf.com/content_ICCV_2019/papers/Ci_Optimizing_Network_Structure_for_3D_Human_Pose_Estimation_ICCV_2019_paper.pdf) (ICCV 2019)

## Requirements
* Python 3.6
* tensorflow (==1.14.0)
* pprint
* prettytable

## Data
Download finetuned Stacked Hourglass detections and our preprocessed H3.6M data (.pkl) [here](https://drive.google.com/drive/folders/1l-Xn5wiDd5ZcnClcqgiBCjHPp4ZjVVsY?usp=sharing) and put them under the directory dataset/. 
If you would like to know how we prepare the H3.6M data, please have a look at the tools/gendb.py.

## Pretrained Models
We provide two kinds of checkpoints which can be downloaded [here](https://drive.google.com/drive/folders/1l-Xn5wiDd5ZcnClcqgiBCjHPp4ZjVVsY?usp=sharing).
* trained with 2d poses detected by finetuned SH
* trained with gt 2d poses

Make two directories experiment/test1 and experiment/test2. Put the GT checkpoint file and SH_DT checkpoint file under test1/ and test2/ respectively, like:
```
${root}/experiment
   └── test1
       └── checkpoints
           ├── best
           └── final
   └── test2
       └── checkpoints
           ├── best
           └── final
```

### Inference and evaluate with the pretrained models.
Inference with GT checkpoint
```
python inference.py --data-type scale --mode gt --test-indices 1 --mask-type locally_connected --knn 3 --layers 3 --in-F 2 --checkpoint best
python evaluate.py --data-type scale --mode gt --test-indices 1
```

Inference with SH_DT checkpoint
```
python inference.py --data-type scale --mode dt_ft --test-indices 2 --mask-type locally_connected --knn 3 --layers 3 --in-F 2 --checkpoint best
python evaluate.py --data-type scale --mode dt_ft --test-indices 2
```

You will get an MPJPE of 32.5mm (GT) and 51.1mm (SH_DT) respectively.


## Train from Scratch
```
python train.py --data-type scale --mode dt_ft --test-indices 3 --mask-type locally_connected --knn 3 --layers 3 --in-F 2
python inference.py --data-type scale --mode dt_ft --test-indices 3 --mask-type locally_connected --knn 3 --layers 3 --in-F 2
python evaluate.py --data-type scale --mode dt_ft --test-indices 3
```

You can also try training and testing with horizontal flip (arg: --flip-data) or confidence values (--in-F 3). Both can bring an extra improvement of about 1mm in MPJPE.

## Citation
If you use this code in your work, please consider citing:
```
@inproceedings{ci2019optimizing,
  title={Optimizing Network Structure for 3D Human Pose Estimation},
  author={Ci, Hai and Wang, Chunyu and Ma, Xiaoxuan and Wang, Yizhou},
  booktitle={Proceedings of the IEEE International Conference on Computer Vision},
  pages={2262--2271},
  year={2019}
}
```

# Acknowledgement
This repo is built on https://github.com/mdeff/cnn_graph#using-the-model and https://github.com/una-dinosauria/3d-pose-baseline.
We would like to thank the authors for publishing their code.
