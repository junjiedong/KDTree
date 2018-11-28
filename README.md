# KDTree
C++ implementation of KDTree &amp; kNN classification on MNIST

This repo implements the KD-Tree data structure that supports efficient nearest neighbor search in k-dimensional vector space in C++, and verifies its functionality by performing kNN classification on the MNIST dataset.

At a high level, KD-Tree (short for k-dimensional tree) is a generalization of a binary search tree that stores points in k-dimensional space. Wikipedia ([k-d tree](https://en.wikipedia.org/wiki/K-d_tree)) has a good brief introduction.

# Files
- `src/Point.h` implements a data structure that encapsulates a point in k-dimensional space
- `src/KDTree.h` is a header-only templatized library that implements `KDTree<size_t N, typename ElemType>`, where N is the dimension of the points, and `ElemType` is the data type of the label associated with the points. The k nearest neighbor search function relies on the Bounded Priority Queue data structure `src/BoundedPQueue.h` implemented by Keith Schwarz
- `src/main.cc` runs basic functionality test on the KD-Tree, and performs kNN classification on the MNIST hand-written digit dataset. By default the code runs with 8 threads and k = 3. The code uses the MNIST data loader `src/mnist.h` written by Nuri Park ([source](https://github.com/projectgalateia/mnist))

# Implementation
The following operations are supported by `src/KDTree.h`
- Efficiently build a balanced KD-Tree from n k-dimensional points (O(nlogn))
- Insert new (point, label) pair into KD-Tree (O(logn))
- Check whether a point is in the KD-Tree (O(logn))
- Retrieve a point in the KD-Tree using the overloaded [ ] operator (O(logn))
- Perform fast k nearest neighbor search `kNNValue(key, k)`
- Deep copy using copy constructor or assignment operator

As one moves down the tree, KD-Tree simply cycles through the axes used to select the splitting planes. When building a subtree from a set of points, KD-Tree uses a linear-time partitioning algorithm to locate the median point, and uses the median point as the splitting pivot so that the resulting tree is balanced. However, if many points are later inserted into the tree, there's no guarantee that the tree remains balanced.

The k nearest neighbor search procedure uses the idea of branch-and-bound. In the procedure, KD-Tree keeps a global set of k best estimates, and performs pruning whenever possible (see [k-d tree](https://en.wikipedia.org/wiki/K-d_tree) for details of the algorithm). However, KD-Tree can be less effective in high-dimensional space due to the curse of dimensionality. Citing Wikipedia: "In high-dimensional spaces, the curse of dimensionality causes the algorithm to need to visit many more branches than in lower-dimensional spaces. In particular, when the number of points is only slightly higher than the number of dimensions, the algorithm is only slightly better than a linear search of all of the points".

# Steps
- Compile `src/main.cc` by running `make` in the root of the repo, and run the generated executable `main`. kNN with k = 3 achieves 97.05% accuracy on the test set (60000 training images, 10000 test images)
