# LANet
```
Our work proposes an adaptive learning attention network (LANet) to solve the problem of color casts and low illumination in underwater images.
```
## Dependencies

The code runs with Python=3.6 and requires Pytorch of version 1.7 or higher. Please `pip install` the following packages:

- `numpy=1.20.2`
- `torchvision=0.8.0`
- `matplotlib=3.4.2`
- `opencv-python=4.5.2.54`
- `scipy=1.7.0`

## Training
```
1. Download the code
2. run Python train.py --input_images-path ./data/trainA/ --label_images_path ./data/trainB/ 
3. Find checkpoint in the "./checkpoints/" folder
The training data includes input data and label data. input data are in the "./data/trainA" folder, label data are in the "./data/trainB" folder
```
## Testing
```
1. pre-trained models in the "./checkpoints/" folder
2. Put your testing images in the "./data/test/" folder 
3. run Python test.py --test_pth ./data/test/ --snapshot_pth ./checkpoints/model_epoch_40.pk
4. Find the result in "./results" folder
```

## Contact
If you have any questions, please contact Shiben Liu at liushiben310@163.com.