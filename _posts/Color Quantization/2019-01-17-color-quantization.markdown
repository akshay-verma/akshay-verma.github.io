---
layout: single
title:  "Color Quantization"
date:   2019-01-19 00:35:10 +0530
categories: computer-vision
toc: true
toc_label: "Content"
header:
  teaser: /assets/images/baboon.png
---

## Introduction

Color quantization is used to reduce the number of colors that are used to represent an image. The objective is to reduce the colors without affecting the visual appearance of the original image. However, some information is lost during the process and hence it is called as a lossy operation.

One of the reason for performing color quantization is to allow rendering of image on hardware supporting limited number of colors. Also, it reduces the space required by the image which in turn allows faster loading on slower devices.

One of the approaches for performing color quantization is using clustering techniques, where colors of the pixel are divided into pre-defined clusters and all the pixels are mapped to the color of the cluster to which they belong. **K-means** is a very popular clustering technique which will be further explored in this post.

## Color Quantization using K-means

K-Means algorithm can be described as follows:

    1. Select randomly k points as cluster centers
    2. Allocate rest of the data points to the closest cluster center
    3. Re-calculate cluster centers, new cluster centers are mean of
       all the points that belongs to a particular cluster
    4. Repeat steps 2 and 3 until the cluster center stops changing

Color quantization using K-means starts by randomly selecting *k* colors from the image as initial cluster centers. Then, the rest of the pixels are assigned to the closest cluster. Once all points are assinged, the center of the clusters are updated to the mean of the pixel colors that belong to a particular cluster. This process is repeated until the cluster center can no longer be updated. At this point, the algorithm has converged to a local solution.

## Analysis

Enough with the theory, let us apply this on few images and see the results!

![Chart](https://github.com/akshay-verma/Computer-Vision/raw/master/color_quantization/chart.png "Comparison of image size after color quantization")

| Name  | Original image size | Size after quantization(K=3)| K=5 | K=10 | K=20 |
| ------------- | ------------- | --------------------------- | --------------------------- | --------------------------- | --------------------------- |
| [baboon.png](https://homepages.cae.wisc.edu/~ece533/images/baboon.png) | 622 KB | 99.4 KB(84% less)|  181 KB(70.9% less) | 299 KB(51.93%) | 402 KB(35.4%) |
| [monarch.png](https://homepages.cae.wisc.edu/~ece533/images/monarch.png) | 599 KB | 73.9 KB(87.66% less) | 128 KB(78.63% less) | 187 KB(68.78%) | 263 KB(56.1%) |
| [tulips.png](https://homepages.cae.wisc.edu/~ece533/images/tulips.png) | 663 KB | 51 KB(92.3% less) |  104 KB(84.3% less)| 186 KB(71.94%) | 311 KB(53.1%) |
|  **Average Compression** |  | 88% |  77.93% | 64.2% | 48.2% |

We can observe that we can achieve almost 50% compression of the original image with K=20 while preserving the visual appearance. There is clear trade-off between reduction in image size and preservation of the original image, and accordingly we can choose the value of K that best suits the purpose.

**Note**: Images used for this analysis are taken from [Public-Domain Test Images for Homeworks and Projects](https://homepages.cae.wisc.edu/~ece533/images/).

## Result

<p style="text-align: center;"><strong>Original image of baboon</strong></p>
<div style="text-align: center;">
  <img src="https://github.com/akshay-verma/Computer-Vision/raw/master/color_quantization/baboon.png"
     alt="Original image of baboon" height="50%" width="50%"/>
</div>

<p style="text-align: center;"><strong>Image after color quantization using 3 clusters (K=3)</strong></p>
<div style="text-align: center;">
  <img src="https://github.com/akshay-verma/Computer-Vision/raw/master/color_quantization/baboon_3.png"
     alt="Original image of baboon" height="50%" width="50%"/>
</div>

<p style="text-align: center;"><strong>Image after color quantization using 20 clusters (K=20)</strong></p>
<div style="text-align: center;">
  <img src="https://github.com/akshay-verma/Computer-Vision/raw/master/color_quantization/baboon_20.png"
     alt="Original image of baboon" height="50%" width="50%"/>
</div>

For more results, check the github repo [here](https://github.com/akshay-verma/Computer-Vision/tree/master/color_quantization).


## Code

The code for main functions is given below. For full code, see [here](https://github.com/akshay-verma/Computer-Vision/blob/master/color_quantization/k_means.py). <br>
Please note that this was an attempt to implement the process from scratch and hence the runtime is high. A better way would be to make use of numpy vectorization.

```python
def performColorQuanitization(img, clusterCenter, K):
    """
    Main method which performs color quantization using K-means

    Args:
        img(str): Location of image on which color quantization is to be performed
        clusterCenter(list): Initial cluster centers (randomly selected)
        K(int): Number of clusters

    Returns:
        clusterCenter(list): Final cluster centers
        classificationVector(list): Indicates to which cluster a pixel belongs to
    """
    error = 1
    while(error > 0):
        classificationVector = classifyImagePoints(img, clusterCenter)
        newClusterCenter = updateColor(img, classificationVector, K)
        error = np.linalg.norm(newClusterCenter - clusterCenter, axis=None)
        clusterCenter = newClusterCenter
    return clusterCenter, classificationVector


def classifyImagePoints(img, clusterCenter):
    """
    Computes distance of each pixel color from the cluster center and assigns pixel
    to the closest cluster

    Args:
        img(str): Location of image on which color quantization is to be performed
        clusterCenter(list): Initial cluster centers (randomly selected)

    Returns:
        classificationVector(list): Indicates to which cluster a pixel belongs to
    """
    classificationVector = {}
    rows, cols = img.shape[:2]
    for row in range(rows):
        for col in range(cols):
            pixel = img[row, col]
            distance = np.linalg.norm(np.uint8([pixel]) - clusterCenter, axis=1)
            classificationVector[(row, col)] = np.argmin(distance)
    return classificationVector


def updateColor(img, classificationVector, K):
    """
    Updates cluster centers by taking mean of the color of the pixel that belongs to the cluster

    Args:
        img(str): Location of image on which color quantization is to be performed
        classificationVector(list): Indicates to which cluster a pixel belongs to
        K(int): Number of clusters

    Returns:
        clusterCenter(list): Updated center of clusters
    """
    clusterCenter = []
    rows, cols = img.shape[:2]
    for clusterNum in range(K):
        points = []
        for row in range(rows):
            for col in range(cols):
                if classificationVector[row, col] == clusterNum:
                    points.append(img[row, col])
        clusterCenter.append(np.round(np.mean(points, axis=0), 2))
        # clusterCenter[clusterNum] = np.round(np.mean(points, axis=0), 2)
    # return np.float32(list(clusterCenter.values()))
    return np.float32(clusterCenter)
```

## Conclusion

We can use color quantization to reduce the memory required by an image without loosing the visual appearance of the original image, which can prove useful on devices with limited memory capacity.
