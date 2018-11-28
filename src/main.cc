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
#include <thread>
#include <mutex>
#include "KDTree.h"
#include "mnist.h"
using namespace std;

using dataset = vector<pair<Point<784>, unsigned int>>;

static const int kNumThreads = 8; // number of threads to use for kNN classification
static int numQueriesProcessed;
static int correctCount;
static mutex queryLock; // lock for global counters

// Perform kNN classification on data[start, end) using kd-tree, and update global counters
static void kNNQueryThread(int start, int end, const KDTree<784, unsigned int>& kd, size_t k, const dataset& data) {
    for (int i = start; i < end; i++) {
        const auto &p = data[i];
        unsigned int pred = kd.kNNValue(p.first, k);
        queryLock.lock();
        ++numQueriesProcessed;
        if (pred == p.second) ++correctCount;
        if (numQueriesProcessed % 500 == 0) cout << numQueriesProcessed << endl;
        queryLock.unlock();
    }
}

// Transform loaded data to the format that KDTree constructor accepts
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

    // Sanity check on the training set
    cout << "Start Sanity Check: contains() should return true for training data, "
            << "and 1-NN training set accuracy should be perfect" << endl;
    bool sanityPass = true;
    for (int i = 0; i < 1000; i++) {
        if (!kd.contains(trainData[i].first) || kd.kNNValue(trainData[i].first, 1) != trainData[i].second) {
            sanityPass = false;
            break;
        }
    }
    if (sanityPass) cout << "Sanity check PASSED!" << endl;
    else cout << "Sanity check FAILED!" << endl;

    // Evaluate performance on test set
    size_t k = 3; // Number of nearest neighbors
    numQueriesProcessed = 0;
    correctCount = 0;
    int queriesPerThread = testCnt / kNumThreads;
    vector<thread> threads;
    cout << "Start evaluating kNN performance on the test set (" << "k = " << k << ")" << endl;
    auto c_start = clock();
    auto t_start = chrono::high_resolution_clock::now();

    for (int i = 0; i < kNumThreads; i++) {
        int start = i * queriesPerThread;
        int end = (i == kNumThreads-1) ? testCnt : start + queriesPerThread;
        threads.push_back(thread(kNNQueryThread, start, end, ref(kd), k, ref(testData)));
    }
    for (thread &t : threads) t.join();

    clock_t c_end = clock();
    auto t_end = chrono::high_resolution_clock::now();
    cout << "Test set accuracy: " << correctCount * 100.0 / testCnt << endl;
    cout << "CPU time elapsed in s: " << (double)(c_end - c_start) / CLOCKS_PER_SEC << endl;
    cout << "Wall time elapsed in s: " << chrono::duration<double>(t_end - t_start).count() << endl;

    free(rawTrainData);
    free(rawTestData);
    return 0;
}
