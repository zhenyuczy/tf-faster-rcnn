## 中文请参考我的[CSDN博客](https://blog.csdn.net/chenzhenyu123456/article/details/81567789)
## 1. Introduction
This is a note about how to use tf-faster-rcnn to train your own model on VOC or other dataset. </br>
My machine and library version: GTX 1060, miniconda 4.5.4, CUDA 9.0, CUDNN 7.1.4, tensorflow-gpu 1.8.0.
## 2. Configuration
This step refers to [https://github.com/endernewton/tf-faster-rcnn](https://github.com/endernewton/tf-faster-rcnn)
### 2.1 Intall opencv, cython, easydict
My version: opencv 3.4.1, cython 0.28.4 and easydict 1.7.
### 2.2 Clone the repository
```linux
git clone https://github.com/endernewton/tf-faster-rcnn.git
```
### 2.3 Build the Cython modules
```linux
cd tf-faster-rcnn/lib
make clean
make
cd ..
```
### 2.4 Install the Python COCO API
```linux
git clone https://github.com/pdollar/coco.git
cd coco/PythonAPI
make
cd ../..
```
### 2.5 Setup data
```linux
cd data
wget http://host.robots.ox.ac.uk/pascal/VOC/voc2007/VOCtrainval_06-Nov-2007.tar
wget http://host.robots.ox.ac.uk/pascal/VOC/voc2007/VOCtest_06-Nov-2007.tar
wget http://host.robots.ox.ac.uk/pascal/VOC/voc2007/VOCdevkit_08-Jun-2007.tar
tar xvf VOCtrainval_06-Nov-2007.tar
tar xvf VOCtest_06-Nov-2007.tar
tar xvf VOCdevkit_08-Jun-2007.tar
ln -s VOCdevkit VOCdevkit2007
cd ..
```
### 2.6 Download pre-trained model
Google Driver: [link](https://drive.google.com/file/d/1KjGKr8A86WkQXzXHisGTgsktxgrH3B_b/view?usp=sharing)</br>
Baidu Cloud: [link](https://pan.baidu.com/s/1LXHHuyYRr0fetnhQtKBayw)</br>
Download the model and put it into the directory `tf-faster-rcnn/`, execute the following command.
```linux
tar xvf voc_0712_80k-110k.tgz
```
### 2.7 Create a folder and a soft link to use the pre-trained model
```linux
NET=res101
TRAIN_IMDB=voc_2007_trainval+voc_2012_trainval
mkdir -p output/$NET/$TRAIN_IMDB
cd output/$NET/$TRAIN_IMDB
ln -s ../../../voc_2007_trainval+voc_2012_trainval ./default
cd ../../..
```
### 2.8 Modify python file
**Modify`tf-faster-rcnn/lib/datasets/voc_eval.py`, 121 line.**
```python
# save
print('Saving cached annotations to {:s}'.format(cachefile))
with open(cachefile, 'w') as f: ---> with open(cachefile, 'wb') as f:
  pickle.dump(recs, f)
```
### 2.9 Demo for testing on custom images
Run the command after modification.
```linux
GPU_ID=0
CUDA_VISIBLE_DEVICES=$GPU_ID ./tools/demo.py
```
### 2.10 End
You can use this model to predict your own dataset(category should be included in the VOC dataset). Of course you may not be satisfied with the mean iou(only 0.20~0.60).
## 3 Train your own model on the VOC dataset
### 3.1 Download CNN pre-trained model
```linux
mkdir -p data/imagenet_weights
cd data/imagenet_weights
wget -v http://download.tensorflow.org/models/resnet_v1_101_2016_08_28.tar.gz
tar -xzvf resnet_v1_101_2016_08_28.tar.gz
mv resnet_v1_101.ckpt res101.ckpt
cd ../..
```
### 3.2 Modify shell file
To speed up the training process, I changed the ITERS in the shell file to **300** here.
#### 3.2.1  `tf-faster-rcnn/experiments/scripts/train_faster_rcnn.sh`
```linux
pascal_voc)
    TRAIN_IMDB="voc_2007_trainval"
    TEST_IMDB="voc_2007_test"
    STEPSIZE="[50000]"
    ITERS=70000 ---> 300 
    ANCHORS="[8,16,32]"
    RATIOS="[0.5,1,2]"
    ;;
```
#### 3.2.2 `tf-faster-rcnn/experiments/scripts/test_faster_rcnn.sh`
```linux
pascal_voc)
    TRAIN_IMDB="voc_2007_trainval"
    TEST_IMDB="voc_2007_test"
    ITERS=70000 ---> 300
    ANCHORS="[8,16,32]"
    RATIOS="[0.5,1,2]"
    ;;
```
### 3.3 Train your model on pascal_voc
For the test script is automatically executed in the train script,  and the test process (4952 images) will take some time. So we only keep the first 200 lines in `data/VOCdevkit2007/VOC2007/ImageSets/Main/test.txt`.
```linux
./experiments/scripts/train_faster_rcnn.sh 0 pascal_voc res101
```
### 3.4 End
Now you can train your own model on the VOC dataset.
## 4 Train your own model on other dataset
This is a dataset for the [car detection competition](http://www.dcjingsai.com/common/cmpt/%E4%BA%A4%E9%80%9A%E5%8D%A1%E5%8F%A3%E8%BD%A6%E8%BE%86%E4%BF%A1%E6%81%AF%E7%B2%BE%E5%87%86%E8%AF%86%E5%88%AB_%E7%AB%9E%E8%B5%9B%E4%BF%A1%E6%81%AF.html). There are two types of files in this dataset 
(see the [dataset](http://www.dcjingsai.com/common/cmpt/%E4%BA%A4%E9%80%9A%E5%8D%A1%E5%8F%A3%E8%BD%A6%E8%BE%86%E4%BF%A1%E6%81%AF%E7%B2%BE%E5%87%86%E8%AF%86%E5%88%AB_%E8%B5%9B%E4%BD%93%E4%B8%8E%E6%95%B0%E6%8D%AE.html) for details). First we decompress the downloaded dataset and put it into a new directory `tf-faster-rcnn/dataset`.
### 4.1 Data Preprocess
#### 4.1.1 Empty directory `data/VOCdevkit2007/VOC2007/`
Run the following command in the directory `tf-faster-rcnn/`.
```linux
rm -rf data/VOCdevkit2007/VOC2007/*
```
#### 4.1.2 `data/VOCdevkit2007/VOC2007/Annotations`
The directory stores the Annotations file (image_name.xml) for all images in the training set.</br>
Here is a [notebook](https://github.com/zhenyuczy/tf-faster-rcnn/blob/master/Annotations.ipynb) to display how to get the .xml files.
#### 4.1.3 `data/VOCdevkit2007/VOC2007/ImageSets/Main`
1. train.txt: all image names in the training set (without .jpg suffix).
2. trainval.txt: all image names in the traing set and validation set.
3. val.txt: all image names in the validation set.
4. test.txt: all image names in the test set(the test script evaluates the mean iou based on it. So you should use all image names of the validation set if test set doesn't have xml files).

**Note: test.txt and val.txt are the same (no .xml file in the test set), simply copy val.txt and rename it.**
#### 4.1.4 `data/VOCdevkit2007/VOC2007/JPEGImages`
All image files (Also you can't put it in that directory if your test set does't have xml files).</br>
Here is a [notebook](https://github.com/zhenyuczy/tf-faster-rcnn/blob/master/ImageSets_and_JPEGImages.ipynb) to display how to get .txt and .jpg files.</br>
#### 4.1.5 End
Move `Annotations`, `ImageSets/Main`, `JPEGImages` under `dataset/` to the directory `data/VOCdevkit2007/VOC2007/`. After that, your dataset is ready.
### 4.2 Modify the program
#### 4.2.1 `lib/datasets/pascal_voc.py`
1. Modify the classes, 36 line
	```python
	self._classes = ('__background__', # always index 0
	                 'car')
	```
2. Remove the "- 1" operation if your dataset is 0-based, 169 line
	```python
	x1 = float(bbox.find('xmin').text)
	y1 = float(bbox.find('ymin').text)
	x2 = float(bbox.find('xmax').text)
	y2 = float(bbox.find('ymax').text)
	```
#### 4.2.2 `lib/datasets/imdb.py`
Modify the program, 105 - 124 line.
```python
def _get_widths(self):
    return [PIL.Image.open(self.image_path_at(i)).size[0]
            for i in range(self.num_images)]

def _get_heights(self):
    return [PIL.Image.open(self.image_path_at(i)).size[1]
            for i in range(self.num_images)]
            
def append_flipped_images(self):
    num_images = self.num_images
    widths = self._get_widths()
    heights = self._get_heights()
    for i in range(num_images):
    	boxes = self.roidb[i]['boxes'].copy()
	oldx1 = boxes[:, 0].copy()
	oldx2 = boxes[:, 2].copy()
	boxes[:, 0] = widths[i] - oldx2 - 1
	boxes[:, 2] = widths[i] - oldx1 - 1
	for ids in range(len(boxes)):
	    if boxes[ids][2] < boxes[ids][0]:
		boxes[ids][0] = 0
	assert (boxes[:, 2] >= boxes[:, 0]).all()
	entry = {'boxes': boxes,
		'gt_overlaps': self.roidb[i]['gt_overlaps'],
		'gt_classes': self.roidb[i]['gt_classes'],
		'flipped': True}
	self.roidb.append(entry)
self._image_index = self._image_index * 2
```
#### 4.2.3 `lib/datasets/voc_eval.py`
27 - 30 line, int() ---> float().
```python
obj_struct['bbox'] = [float(bbox.find('xmin').text),
                      float(bbox.find('ymin').text),
                      float(bbox.find('xmax').text),
                      float(bbox.find('ymax').text)]
```
### 4.3 Train your model
#### 4.3.1 Training
1. Remove old model
	```linux
	rm -rf output
	```
2. Clear cache
	```linux
	rm data/cache/voc_2007_test_gt_roidb.pkl
	rm data/cache/voc_2007_trainval_gt_roidb.pkl
	rm data/VOCdevkit2007/VOC2007/ImageSets/Main/test.txt_annots.pkl
	```
3. Update parameters</br>
  3.1 Modify stepsize and iteations in `experiments/scripts/train_faster_rcnn.sh`.</br>
  3.2 Modify the parameters of the model in `experiments/cfgs`. </br>
  3.3 Modify the training and test parameters in `lib/model/config.py`.</br>
4. Training
	```linux
	./experiments/scripts/train_faster_rcnn.sh 0 pascal_voc res101
	```
**Note:** Make sure that the ITERS in `train_faster_rcnn.sh` and `test_faster_rcnn.sh` are same.
#### 4.3.2 Predict
Use `./tools/demo.py` to predict your test images. This is an [demo example](https://github.com/zhenyuczy/tf-faster-rcnn/blob/master/demo.py) on my github. A simple explanation of the input: </br>
  1. **demo_net**: classification network architecture.</br>
  2. **demo_ite**: ITERS of the network.</br>
  3. **demo_dir**: The test set directory.</br>
  4. **demo_vis**: Whether to visualize the test image.</br>
  5. **write_csv**: Whether to write the predicted boxes to the csv file.</br>
  6. **dataset**: Select the dataset format.</br>

Note: In order to observe the prediction performance better, you can first put a few test images in `data/demo`, and set demo_vis to True when executing. If you are satisfied with the visualization results, predict all the test images after modify `demo_dir` and set `demo_vis` to False (** Visualizing many images at the same time is terrible, remember!!!**).
### 4.4 How to improve the performance of your model
It is critical to understand all the procedures in this project. Only then can you train your network by modifying parameters, adding data augmentation and choosing different metrics.
