/**
 * File: main.cc
 * ------------------------
 * Use KD-Tree to perform hand-written digit classification on the MNIST dataset
 * Training set: 60000 images
 * Test set: 10000 images
 */

#define USE_MNIST_LOADER
#define MNIST_DOUBLE

#include <iostream>
#include <vector>
#include <time.h>
#include "KDTree.h"
#include "mnist.h"
using namespace std;

using dataset = vector<pair<Point<784>, unsigned int>>;

void transformData(mnist_data* rawData, unsigned int cnt, dataset& data) {
    for (int idx = 0; idx < cnt; idx++) {
        std::vector<double> tmp;
        for (int i = 0; i < 28; i++) {
            for (int j = 0; j < 28; j++) {
                tmp.push_back(rawData[idx].data[i][j]);
            }
        }
        Point<784> newPoint;
        copy(tmp.begin(), tmp.end(), newPoint.begin());
        data.push_back(make_pair(newPoint, rawData[idx].label));
    }
}

int main(int argc, char **argv) {
    // Load the MNIST dataset
    mnist_data *rawTrainData;
    unsigned int trainCnt;
    mnist_data *rawTestData;
    unsigned int testCnt;
    mnist_load("mnist_data/train-images-idx3-ubyte", "mnist_data/train-labels-idx1-ubyte", &rawTrainData, &trainCnt);
    mnist_load("mnist_data/t10k-images-idx3-ubyte", "mnist_data/t10k-labels-idx1-ubyte", &rawTestData, &testCnt);

    cout << "Finished loading data from disk!" << endl
        << "Training set size: " << trainCnt << endl
        << "Test set size: " << testCnt << endl;

    // Transform the loaded data to vector<pair<Point<784>, unsigned int>>
    dataset trainData;
    dataset testData;
    transformData(rawTrainData, trainCnt, trainData);
    transformData(rawTestData, testCnt, testData);
    cout << "Finished transforming dataset!" << endl;

    // Construct KD-Tree using training set
    KDTree<784, unsigned int> kd(trainData);
    cout << "Finished building KD-Tree!" << endl;

    // Evaluate performance on test set
    int correctCount = 0;
    size_t k = 1; // Number of nearest neighbors
    cout << "Start evaluating kNN performance on the test set (" << "k = " << k << ")" << endl;
    clock_t start = clock();
    for (int i = 0, len = testData.size(); i < len; i++) {
        const auto &p = testData[i];
        unsigned int pred = kd.kNNValue(p.first, k);
        if (pred == p.second) ++correctCount;
        if ((i+1) % 500 == 0) cout << i+1 << endl;
    }
    cout << "Test set accuracy: " << correctCount * 100.0 / testData.size() << endl;
    cout << "Time elapsed in s: " << (double)(clock() - start) / CLOCKS_PER_SEC << endl;

    free(rawTrainData);
    free(rawTestData);
    return 0;
}
