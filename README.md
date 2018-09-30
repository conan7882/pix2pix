# pix2pix 
- Tensorflow implementation of [
Image-to-Image Translation with Conditional Adversarial Networks](https://arxiv.org/abs/1611.07004) (CVPR 2017)
- The conditional adversarial network is trained to learn the image-to-image mapping.
- The generator is a U-Net (a encoder-decoder with skip layers), since the input and output may share low level features for some tasks (i.e. image colorizaton).
- The discriminator is used to classify if a image patch is real or fake (PatchGAN). Because L1 loss forces the low-frequency correctness (blurred images) and it is sufficient for discriminator to focus on high-frequency structure locally. Then the output of discriminator is averaged over all the patches in the image. Another advantage of using PatchGAN is that it has fewer parameters than full size GAN and can be applied to arbitrary size images.
- Additional L1 loss is added to generator loss to make the generated image near the groundtruth. L1 loss is chosen because it is less encourage blurring images than L2 loss as mentioned in section 3.1 of the paper.

## Requirements
- Python 3.3+
- [Tensorflow 1.9+](https://www.tensorflow.org/)
- [Numpy](http://www.numpy.org/)
- [Scipy](https://www.scipy.org/)

## Implementation Details
- The network architecture is the same as the paper. U-Net decoder is used for generator and 70 x 70 distriminator is used. Dropout with probabilty 0.5 is used after each transpose convolutional layers in decoder of generator both for training and testing. 
- Generator is updated by the loss GAN_loss + 100 * L1_loss.
- All the images are normalized to range [-1, 1] before fed into the network. For facades and maps dataset, when training, images are first rescaled to 286 x 286 and then randomly mirrored and cropped to 256 x 256. For shoes dataset, images are rescaled to 256 x 256 for both training and testing.
- Weights are initialized from a Gaussian distribution with mean 0 and standard deviation 0.02. Learning rate is set to be 2e-4. Facades and maps dataset are trained for 200 epochs, while shoes dataset is trained for 20 epochs.


## Result on Validation Set
### Facades
*No.* | *Domain A* | *Domain B* | *Output A* |
:-- | :---: | :---: |:---: |
1|<img src = 'figs/facades/cmp_x0001.jpg' height = '200px' width = '200px'> | <img src = 'figs/facades/cmp_x0001.png' height = '200px' width = '200px'> | <img src = 'figs/facades/im_0000.png' height = '200px' width = '200px'> |
2|<img src = 'figs/facades/cmp_x0004.jpg' height = '200px' width = '200px'> | <img src = 'figs/facades/cmp_x0004.png' height = '200px' width = '200px'> | <img src = 'figs/facades/im_0003.png' height = '200px' width = '200px'> |
3|<img src = 'figs/facades/cmp_x0005.jpg' height = '200px' width = '200px'> | <img src = 'figs/facades/cmp_x0005.png' height = '200px' width = '200px'> | <img src = 'figs/facades/im_0004.png' height = '200px' width = '200px'> |
4|<img src = 'figs/facades/cmp_x0006.jpg' height = '200px' width = '200px'> | <img src = 'figs/facades/cmp_x0006.png' height = '200px' width = '200px'> | <img src = 'figs/facades/im_0005.png' height = '200px' width = '200px'> |
5|<img src = 'figs/facades/cmp_x0008.jpg' height = '200px' width = '200px'> | <img src = 'figs/facades/cmp_x0008.png' height = '200px' width = '200px'> | <img src = 'figs/facades/im_0007.png' height = '200px' width = '200px'> |
6|<img src = 'figs/facades/cmp_x0009.jpg' height = '200px' width = '200px'> | <img src = 'figs/facades/cmp_x0009.png' height = '200px' width = '200px'> | <img src = 'figs/facades/im_0008.png' height = '200px' width = '200px'> |
7|<img src = 'figs/facades/cmp_x0010.jpg' height = '200px' width = '200px'> | <img src = 'figs/facades/cmp_x0010.png' height = '200px' width = '200px'> | <img src = 'figs/facades/im_0009.png' height = '200px' width = '200px'> |
8|<img src = 'figs/facades/cmp_x0007.jpg' height = '200px' width = '200px'> | <img src = 'figs/facades/cmp_x0007.png' height = '200px' width = '200px'> | <img src = 'figs/facades/im_0006.png' height = '200px' width = '200px'> |
9|<img src = 'figs/facades/cmp_x0002.jpg' height = '200px' width = '200px'> | <img src = 'figs/facades/cmp_x0002.png' height = '200px' width = '200px'> | <img src = 'figs/facades/im_0001.png' height = '200px' width = '200px'> |
10|<img src = 'figs/facades/cmp_x0003.jpg' height = '200px' width = '200px'> | <img src = 'figs/facades/cmp_x0003.png' height = '200px' width = '200px'> | <img src = 'figs/facades/im_0002.png' height = '200px' width = '200px'> |

### Shoes

*Domain A and B* | *Output B* | *Domain A and B* | *Output B* |
:---: | :---: |:---: | :---: |
<img src = 'figs/shoes/1_AB.jpg' height = '130px'> | <img src = 'figs/shoes/2/im_1_AB.png' height = '130px'> | <img src = 'figs/shoes/4_AB.jpg' height = '130px'> | <img src = 'figs/shoes/2/im_4_AB.png' height = '130px'> |
<img src = 'figs/shoes/39_AB.jpg' height = '130px'> | <img src = 'figs/shoes/2/im_39_AB.png' height = '130px'> | <img src = 'figs/shoes/24_AB.jpg' height = '130px'> | <img src = 'figs/shoes/2/im_24_AB.png' height = '130px'> |
<img src = 'figs/shoes/51_AB.jpg' height = '130px'> | <img src = 'figs/shoes/2/im_51_AB.png' height = '130px'> | <img src = 'figs/shoes/174_AB.jpg' height = '130px'> | <img src = 'figs/shoes/2/im_174_AB.png' height = '130px'> |
<img src = 'figs/shoes/181_AB.jpg' height = '130px'> | <img src = 'figs/shoes/2/im_181_AB.png' height = '130px'> | <img src = 'figs/shoes/194_AB.jpg' height = '130px'> | <img src = 'figs/shoes/2/im_194_AB.png' height = '130px'> |
<img src = 'figs/shoes/187_AB.jpg' height = '130px'> | <img src = 'figs/shoes/2/im_187_AB.png' height = '130px'> | <img src = 'figs/shoes/198_AB.jpg' height = '130px'> | <img src = 'figs/shoes/2/im_198_AB.png' height = '130px'> |

### Maps

*Domain A and B* | *Output A* | *Output B*
:---: | :---: | :---: |
<img src = 'figs/maps/1.jpg' height = '210px'> | <img src = 'figs/maps/2/im_1.png' height = '210px'> | <img src = 'figs/maps/1/im_1.png' height = '210px'>
<img src = 'figs/maps/100.jpg' height = '210px'> | <img src = 'figs/maps/2/im_100.png' height = '210px'> | <img src = 'figs/maps/1/im_100.png' height = '210px'>
<img src = 'figs/maps/15.jpg' height = '210px'> | <img src = 'figs/maps/2/im_15.png' height = '210px'> | <img src = 'figs/maps/1/im_15.png' height = '210px'>
<img src = 'figs/maps/14.jpg' height = '210px'> | <img src = 'figs/maps/2/im_14.png' height = '210px'> | <img src = 'figs/maps/1/im_14.png' height = '210px'>
