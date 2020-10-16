# Notes on [OverFeat: Integrated Recognition, Localization and Detection using Convolutional Networks](https://arxiv.org/pdf/1312.6229.pdf)

###### tags: `notes` `supervised learning` `object detection` `overfeat`

## Brief Outline

---

This paper presents a framework for classification, localization and detection using a multiscale and sliding window approach. It can do mutiple tasks using a single shared network. Second important findings of this paper is explaining how ConvNets can be effectively used for detection and localization tasks.


## Introduction

---
* The performance of ConvNets is just decent for small datasets, whereas the larger datasets have enabled them to improve the SOTA on such datasets.
* One of the major advantage of ConvNets is that they alleviate the requirement of manual design of feature vectors. But it being labelled data hungry is one of it's disadvantage.
* This paper presents that a single ConvNet can do object detection, recognition, and localization.
* It also demonstrates a method to localize and detect by y accumulating predicted bounding boxes and by doing this detection can be performed without training on background samples.
* State of the art results were obtained on the ILSVRC 2013 localization and detection tasks.
* The images choosen from the ImageNet classification dataset might contain objects of interests in various sizes and positions in the image, the authors had 3 ideas to address this.
* The first idea is using a ConvNet based sliding window approach at multiple locations, but this lead to poor localization and detection.
* The second idea is to do bounding box regression along with classification. 
* The third idea is to accumulate the evidence for each category at each location and size.
* Several authors tried to localise the objects using the ConvNets such as [Osadchy et al.](https://link.springer.com/chapter/10.1007/11957959_10) used for simultaneous detection and pose estimation and  [Taylor et al.](https://papers.nips.cc/paper/4143-pose-sensitive-embedding-by-nonlinear-nca-regression) uses a ConvNet to estimate the location of body parts (hands, head, etc) so as to derive the human body pose.
* [One approach](https://ieeexplore.ieee.org/document/4408909) involves training the ConvNet to classify the central pixel of its viewing window as a boundary between regions or not.
* The approach of training the ConvNet to classify the central pixel of the viewing window with the category of the object it belongs to has a disadvantage that it requires dense pixel-level labels for training.
* To address this, an classification method can be used at the optimal location in the search space which increases recognition accuracy by reducing unlikely object regions.
* In the paper, the authors also claim that their paper is  the first one to provide this application of ConvNets which localization and detection for ImageNet data. 

## Vision Tasks
* There are three tasks, performed using a single framework:
   1. Classification
   2. Localization
   3. Detection
* All the results are reported on the 2013 ImageNet Large Scale Visual Recognition Challenge (ILSVRC2013).
* Classification task: Five guesses are allowed to guess the main object in the image.
* Localization task: Five guesses to predict the bounding box for the main object and to be correct it should have atleast 50% IOU with the the groundtruth box.
* False positives are penalized by mAP.
* The below given image shows the detection and localization, 
![](https://i.imgur.com/cSMOLfV.png)

## Classification
* The architecture of the neural network is an improvised version the best ILSVRC12 architecture by [Krizhevsky et al.](https://papers.nips.cc/paper/4824-imagenet-classification-with-deep-convolutional-neural-networks.pdf) popularly known as AlexNet.

### Model design and training
* The model is trained on the ImageNet dataset, the input size of the image is same as the one proposed in [Krizhevsky et al.](https://papers.nips.cc/paper/4824-imagenet-classification-with-deep-convolutional-neural-networks.pdf) during training.
* Every image is first downsampled to 256 pixels and then 5 random crops (and their horizontal flips) of size 221x221 pixels are extracted and presented to the network.
* The weights in the network are initialized randomly with \\((µ, σ) = (0.1 × 10{^−2})\\) and the learning is done using the SGD with a momentum term of 0.6 and an ℓ2 weight decay of \\(1 × 10^{−5}\\).
* The learning rate is initially 5 × 10−2 and is successively decreased by a factor of 0.5 after a certain number of epochs and a dropout rate of 0.5 is applied on the fully connected layers.
* Basically the 1-5 layers are similar to the AlexNet but with i) zero contrast normalization ii) zero overlapping in the pooling layers iii) has a larger stride initially.
* Spatial size of the feature maps varies during the test time as we downsample images.
* The fully connected layers are implemented as sliding window approach during test time which are \\(1X1\\) convolutions.

### Feature Extractor
* Two models of feature extractors are provided in this paper named overfeat.
* One model for accuracy and the other for its speed (in comparison to each other).
* The architectures of both the models are listed in the tables below, they mainly vary in the stride, number of stages and the feature maps.
* The below table is for the fast model,
![](https://i.imgur.com/HYrbfL0.png)
* The below table is for the accurate model,
![](https://i.imgur.com/SCQPruj.png)

### Multi scale classification
* In the AlexNet paper, multi-view voting is used to boost performance but this approach may miss out some features and also some views might overlap.
* Moreover, there prefer a the input image in a single scale which might not be the optimal one for the the network.
* In this paper, multiple scales of input images are taken with the classical sliding window approach of the ConvNets, the result would be different spatial feature maps.
* The total subsampling ratio is 36, which is also called as effective stride meaning this architecture can only produce a classification vector every 36 pixels in the input dimension along each axis.
* To decrease the subsampling ratio, the paper follows the approach presented in [Giusti et al.](https://arxiv.org/pdf/1302.1700.pdf), and apply the last subsampling operation at every offset, this reduces it to subsampling ratio of 12.
* In the paper, 6 different scale inputs are used, which results in 5 unpooled maps.
* After pooling the following steps are taken:
 (a) For a single image, at a given scale, there are 5        unpooled feature maps.
 (b) Each feature map goes through \\(3X3\\) pooling, replicated pooled maps are produced.
 (c ) For the further layers, classifier is applied as a sliding window.
 (d) The output maps are reshaped into 3D output map.
 The process is explained in the case of 1Dim,
 ![](https://i.imgur.com/cNoKCuy.png)
 * The same is done for horizontally flipped versions and the results are reported as:
 (a) Taking the spatial max for each class, at each scale        and flip.
 (b) Averaging across C-dimensions.
 (c ) Taking the top elements from mean class vectors.
 * At an intuitive level, the first 5 layers work in a different way compared to the rest of the model.
 * In the first part, the filters are convolved across the  entire image in one pass which is efficient than the sliding window approach.
 * Thus the classifier has a fixed-size 5x5 input. The exhaustive pooling scheme (with single pixel shifts (∆x, ∆y)) ensures that we can obtain fine alignment between the classifier and the representation of the object in the feature map.

### Results
* The number of scales used have direct impact on the performance.
* The below given table demonstrates this fact,
![](https://i.imgur.com/skRLsFa.png)
* The below given table describes the test set results reported in top-5 error rate,
![](https://i.imgur.com/QSPMLmb.png)

### ConvNets and sliding window efficiency
* ConvNets are efficient compared to many other sliding window approaches as they have shared computations.
* The convolution is applied to the entire image and the varying input sizes produce varying spatial outputs as shown below,
![](https://i.imgur.com/OJKgyBk.png)
* During the training process the end layers are fully connected layers but during test time they are \\(1X1\\) convolutions.

## Localisation
Similar to classification the localisation is done by bounding box regression, the combination of the results is done as follows,

### Generating predictions
* The regression is done after classification on the same feature vector.
* The output of the final softmax layer gives the confidence score of an object being present at that particular location.

### Regression training
* The pooled feature maps are obtained from layer 5.
* The network has 2 fully-connected hidden layers of size 4096 and 1024 channels, respectively and the final output layer has 4 units for 4 coordinates of the bounding box.
* The clear understanding can be obtained from the following image 
![](https://i.imgur.com/L0BwOGY.png)
- They train the regression network using an $ℓ_2$ loss between the predicted and true bounding box for each example. The final regressor layer is class-specific, having 1000 different versions, one for each class.
- Authors observed that training on a single scale performs well on that scale and still performs reasonably on other scales. However, training multi-scale makes predictions match correctly across scales and exponentially increases the confidence of the merged predictions.

### Combining predictions
The individual predictions are combined via a greedy strategy applied to the regressor bounding boxes, using the following algorithm :
1. Assign to $C_s$ the set of classes in the top $k$ for each scale $s$ ∈ 1 . . . 6, found by taking the maximum detection class outputs across spatial locations for that scale.
2.  Assign to $B_s$ the set of bounding boxes predicted by the regressor network for each class in $C_s$, across all spatial locations at scale $s$.
3. Assign $B \leftarrow \cup_s\  B_s$
4. Repeat merging until done:
    - ($b^*_1 , b^∗_2$ ) = $\underset{b1\ \neq\ b2\ ∈\ B}{argmin}$ *match_score*($b_1, b_2$)
    - If match_score$(b_1, b_2) > t$, stop.
    - Otherwise, set $B \leftarrow B/{(b^∗_1 , b^∗_2)}\  \cup$ *box_merge*($b^*_1 , b^∗_2$)

In the above, we compute *match_score* using the sum of the distance between centers of the two bounding boxes and the intersection area of the boxes. *box_merge* compute the average of the bounding boxes’ coordinates.
The final prediction is given by taking the merged bounding boxes with maximum class scores. This is computed by cumulatively adding the detection class outputs associated with the input windows from which each bounding box was predicted.
 

### Experiments
This method was the winner of the 2013 localization competitions with 29.9% error. The multiscale and multi-view approach was critical to obtaining good performance. Combining regressor predictions from all spatial locations at >1 scales gives vastly better error rates.

### Detection
- Detection training is similar to classification training but in a spatial manner. Multiple location of an image may be trained simultaneously. The main difference with the localization task, is the necessity to predict a background class when no object is present.
- Traditionally, negative examples are initially taken at random for training. Then the most offending negative errors are added to the training set in bootstrapping passes. 
- Independent bootstrapping passes render training complicated and risk potential mismatches between the negative examples collection and training times. Additionally, the size of bootstrapping passes needs to be tuned to make sure training does not overfit on a small set. 
- To circumvent all these problems, we perform negative training on the fly, by selecting a few interesting negative examples per image such as random ones or most offending ones. This approach is more computationally expensive, but renders the procedure much simpler





