# Notes on [Augmented Autoencoders: Implicit 3D Orientation Learning for 6D Object Detection](https://arxiv.org/pdf/1902.01275.pdf)

tags: `notes` `semi-supervised` `domain randomization` `denoising autoencoder`

**Authors**
[Jayesh Kandpal](https://github.com/jayeshk7)
[Aayush Fadia](https://github.com/aayush-fadia)
[Saketh Bachu](https://github.com/sakethbachu)

A brief outline
---
This paper presents a real-time RGB-based pipeline for object detection and 6D pose estimation, based on a variant of denoising autoencoder, which is an augmented encoder trained on views of a 3D model using domain randomization.


## Introduction
* Estimating an object's pose in six degrees of freedom (translation and rotation) is necessary for robots to be able to grasp/manipulate an object, and this information is also crucial in augmented reality.
* The existing methods often are not robust against object occlusion, background clutter, object symmetry, etc and also are not ideal for real time applications.
* This paper solves the 6D pose estimation challenge without requiring a large-scale dataset, and is able to do so in real-time. It addresses the problems mentioned and proposes a pipeline to estimate the 6D pose with the 2D object detection as its first part and then estimating the 3D pose using an augumented autoencoder trained on multiple views of the rendered objects.
* Further, the 3D translation \\(\hat t_{obj2cam}\\) is calculated the resulting  euclidean transformation \\(\hat H_{obj2cam}\\)  is refined using ICP.


![](https://i.imgur.com/jtPAxJ6.png)



## Simulation to Reality Transfer

### Photo-realistic rendering
* The photo realistic rendering of objects views and backgrounds might be beneficial for some tasks like object detection and is apt for simple environments. However, photo-realistic modelling is often imperfect and requires much effort.


### Domain adaptation (DA)
* The key idea is to leverage data from a source domain to a domain of which only a small portion of labeled data or unlabeled data is available.
* GANs have been deployed for unsupervised DA by generating realistic image from synthetic image, but they often yield fragile results. Supervised DA requires much lower annotated data.

### Domain randomization
* It is based on the hypothesis that the model will generalize to real images if trained on a variety of augmented semi-realistic ones (augmented with random lighting conditions, backgrounds, saturation, etc).
* This strategy has been previously used in tasks like 3D shape detection and it generalizes well for real images, especially when the renderings are almost photo-realistic.

---

## Training Pose Estimation with SO(3) Targets

Difficulties of training with fixed SO(3) parameterizations :
1. #### Regression
    -  The idea of regressing on fixed SO(3) parameterization like quaternions seems natural but can have convergence issues due to pose ambiguities.
    - Also, applying perspective-n-point algorithm on 2D-3D regressions does not eliminate the pose ambiguity.

2. #### Classification
    * Discretization of SO(3) is not a viable option as even a coarse interval of 5$^{\circ}$ will create 50,000 classes and each class is generally sparse.
    *  Representing 3D orientation by discretizing viewpoint and in-plane rotation was used in [Kehl et al., 2017](https://arxiv.org/abs/1711.10006) which gave ambiguous class combinations to non-canonical views.

3. #### Symmetries
    * Symmetries cause pose ambiguities and these cause one-to-many mappings which disturb the learning process.
    * Some methods which were previously used include, ignoring one axis of rotation, training of an extra CNN to predict symmetries, etc are tedious ways to filter out object detection and need extra work to address other problems too.
 ![](https://i.imgur.com/nsdnYNO.png)


## Learning Representations of 3D orientations

### Descriptor Learning
* The approach presented by [Wohlhart and Lepetit., 2015](https://arxiv.org/abs/1502.05908) which uses triplet loss relies on pose-annotated sensor data and is not immune against symmetries.
* The loss can cause problems in classifying different orientations of an obeject having similar appearance.
* The work presented by [Balntas et al., 2017](https://openaccess.thecvf.com/content_ICCV_2017/papers/Balntas_Pose_Guided_RGBD_ICCV_2017_paper.pdf) acknowledge the problem of symmetries by correlating the descriptor and pose distances.
* The paper presents an approach based on an augumented autoencoder which deals only with the reconstructed views thus by addressing the symmetry problem inherently and assigning 3D orientation is done after training.
* One similar method presented by [Kehl et al., 2016](https://arxiv.org/abs/1607.06038) isn't efficient during the test time as it needs compare a lot of patches and also ignores holistic relations between object features.

---

## Method
### Outline
The pipeline is as follows:
* Object bounding box detected by any object detector.
* The cropped object is used to estimate 3D translational position of the object.
* The 3-D orientation is estimated from the cropped region of the object.

The object detection algorithm is not covered in this paper.
For learning to estimate rotational pose, the algorithm uses an autoencoder structure, that learns to generate  clean images of an object, given a *synthetically augmented* image of the object. This clean, generated image must be of the same orientation as the original, noisy images. This means that the bottleneck layer of the autoencoder will retain information about the orientation of the object, and discard everything else, since only orientation information is needed for reconstructing a clean image of that orientation.
During test time, the encoder is used to get the latent vector, that encodes information about the orientation.

### Proof of concept, smaller scale experiment
Before moving to a real, 3D object, the method was tested on a dataset of white squares on a black background. This diagram illustrates this.

![](https://i.imgur.com/XEQvrvv.png =1100x)

- The graphs have rotation angle on the X-axis, and L1 and L2 norms of the latent vector along the Y-axis. The aim is to find a co-relation between the orientation of the object and these norms of the latent vector. If a pattern exists between the two, we know that the latent vector encodes information about the orientation. 
- The first graph was created by training an autoencoder with source images from (a) and corresponding target images from (a). The graph shows no relation between the latent vector and the square's rotation. 
- The bottom graph, though, is a different story. Here, source images were taken from (d), where images were of random scales, rotations and translations. The autoencoder was then trained to generate the corresponding image from (a), which had only varying rotation. 
- Here, decoder only has to have rotation information to be able to generate the correspondiung target. This is why, the latent dims only encode information about the rotation, and throws away the rest, because it is not needed. And, we see this in the graph, there is a very strong co-relation between the angle and the hidden dims, it has a phase of 90 degrees, because a square will look the same when rotated by 90 degrees. 

This small experiment shows that this can, in fact, be done. The authors now move to implementing the same thing on real-world objects.

### On to Real world objects
* Now, to apply this on the real world, we capture images of the target object from multiple angles and orientations, we take care to keep it at the same distance from the camera.

* This is analogous to (a) in the proof of concept. Now, we need to generate the set (d). For this, we simply have to augment the images in (a) in ways that will be anticipated in the real world. This is analogous to scaling and translating the images in (a) to make the ones in (d). We can do this by any of the three methods mentioned above.

* Now, while training, we simply make the autoencoder generate the clean images with rotation only (a) from the augmented images (d). Exactly as we did for the smaller scale proof of concept. The reasons are all the same, since only orientation information is required to generate clean image in that orientation, that's all the network will deposit in the latent vector.

* The architecture of this network is shown below. The loss function, between the decoder generated image and the target image is a bootstrapped MSE loss, which is basically found by taking into account only those pixels that have the greatest Squares Difference, or are most different. This prevents the network from settling into a local minima, like generating full black images. 

![](https://i.imgur.com/kyEju5T.png)




### Inferring rotation from latent vector
* After training is done, we simply evaluate latent vectors of each image in our train set. 
* Since we also know the orientations for these images, we can store them in a dictionary of (latent vector ->orientation) pairs. 
* Then, while being used, the decoder generates a latent vector for a view of an object in an unknown orientation. 
* We simply compare this vector with all the ones we have in the dictionary from the training set, and return the orientation corresponding to the closest vector (by cosine similarity) from the training set.

### Projective Distance Estimation
* We know the bounding box of our object from the 2D object detector. It stands to logic, that the farther away the object is, the smaller it's bounding box will be. 
* While collecting the training set, we store the camera's focal length, the distance of the object from the camera, and the diagonal length of the bounding box of the object in the training image. 
* When it is deployed, we will have the bounding box of the real world image, and thus, the length of it's diagonal. We will also know the focal length of this test camera. 
* Then, using the pinhole camera model, we can calculate the distance as: $$\hat t_{real,\ z} = t_{syn,\ n} * \frac{||bb_{syn,\ argmax(cos_i)}||}{||bb_{real}||} * \frac{f_{real}}{f_{syn}}$$
* Thus, we have solved the scale ambiguity. 
* Now, it follows that: $$\Delta = \hat t_{real,\ z}\ K^{-1}_{real}\ bb_{real,\ c} - t_{syn,\ z}\ K^{-1}_{syn}\ bb_{syn,\ c}$$
$$ \hat t_{real} = t_{syn} + \Delta\hat t$$
* Where \\(\Delta \hat t\\) is the estimated vector. 
* The  \\(K_{real},\ K_{syn}\\) are the camera matrices.
* The \\(bb_{real,c},\ bb_{syn,\ c}\\) are the bounding box centers in homogeneous coordinates.
* The \\(\hat t_{real},\ t_{syn}\\) are the translation vectors from camera to object centers.
 
### Perspective Correction

* The objects placed in the same orientation will look different when they are translated across the image.
* An object looked at directly, such that it falls in the center of the image, will look different than an object in the same orientation, but at the edge of the image, not pointed at dirtectly by the camera. 
* This method did have inaccuracies when the object was placed near the edges of the image, which can be corrected by determining the object rotation that approximately preserves the appearance of the object,

$$ {\ \alpha_{x}\choose \alpha_{y}} = {-arctan(\hat t_{real,\ y}/\ \hat t_{real, \ z}) \choose arctan(\hat t_{real,\ x}/\ \sqrt{(\hat t^2_{real,\ z}+\hat t^2_{real,\ y})}}$$
$$ \hat R_{obj2cam} = R_{y}(\alpha_{y})R_{x}(\alpha_{x})\hat R'_{obj2cam}$$
* Where \\(\alpha_{x},\ \alpha_{y}\\) are the angles around the camera axes, and \\(R_{y}(\alpha_{y}),\ R_{x}(\alpha_{x})\\) are the corresponding rotational matrices.
* This technique gives a boost in accuracy.

###  Inference Time
* The optimal performance at real time is given by the RGBbased pipeline is real-time capable at ∼42Hz on a Nvidia GTX 1080.
* This also leaves room for tracking algorithms and dealing with multi objects is feasible.
---
## Evaluation

Authors trained the AAEs on the reconstructed 3D models of T-LESS dataset, except for objects 19-23 where they trained on the CAD models because the pins were missing in the reconstructed plugs. They noticed, that the geometry of some 3D reconstruction in T-LESS is slightly inaccurate which badly influences the RGB-based distance estimation since the synthetic bounding box diagonals are wrong. Therefore, in the second training run they only train on the 30 CAD models.

### Metrics
- As in the SIXD challenge, the authors report the recall of correct 6D object poses at $err_{vsd} < 0.3$ with tolerance $τ = 20mm$ and $> 10 \%$ object visibility, where $err_{vsd}$ is Visible Surface Discrepancy, an ambiguity-invariant pose error function that is determined by the distance between the estimated and ground truth visible object depth surfaces.

### Ablation study 
- To assess the AAE alone, only the 3D orientation of object 5 from the T-LESS dataset was tested. The effect of different color augmentations was found to be cumulative.
- The authors found that training with real object recordings provided in T-LESS with random Pascal VOC background and augmentations yields only slightly better performance than training with synthetic data.
- Performance started to saturate for *latent dimensions $\geq$ 64*.

### Results
- To reach the performance of the state-of-the-art depth-based methods a depth based ICP was necessary to refine the estimates.
- This pipeline outperforms all 15 reported T-LESS results on the 2018 BOP benchmark. The RGB-only results of this algorithm can compete with previous RGB-D learning based approaches. Previous SOTA approaches perform a time consuming refinement search through multiple pose hypotheses while here they only perform ICP on a single pose hypotheses.
- Results show that domain randomization helped generalize from 3D reconstructions and untextured CAD models as long as the considered object is not higly textured. 
- However, it does not outperform the current SOTA on the LineMOD dataset. The authors pointed out multiple issues : 
    1. The real training and test set are strongly correlated and approaches using real training set can overfit. 
    2. The advantage of not suffering from pose ambiguities does not matter much in LineMOD because most object views are pose-ambiguity free.
    3. They train and test poses from the whole SO(3) as opposed to only a limited range in which the test poses lie.


### Failure cases 
- Most of the errors arised due to strong occlusions and missed detections. The dependence of the object distance on the bounding box seems to be a weak point, as under occlusion the bounding box prediction is inaccurate.
- Furthermore, on strongly textured objects the AAEs should not be trained without rendering the texture since otherwise the texture might not be distinguishable from shape at test time. Also, the sim2real transfer for strongly reflective objects can be challenging. - first sentence is doubt
- Some objects like long, thin pens can fail because their tight object crops at training and test time appear very near from some views and very far from other views, thus hindering the learning of proper pose representations.

### Rotation and translation histograms
- To investigate the effect of ICP, rotation and translation error histograms were plotted of 2 T-LESS objects.
- They found that the translation error was strongly improved through the depth-based ICP while the rotation estimate was hardly refined. 
- The projective distance estimation fails to produce accurate distance predictions for partly occluded objects, as the bounding boxes can become less accurate.


### Hardware demonstrations

The presented AAE when ported onto a nvidia jetson tx2, together with a small footprint mobilenet, detected and estimated pose of the objects at over 13Hz. The algorithm also showed robustness against different lighting condition.
