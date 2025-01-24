#include <opencv2/core.hpp>
#include <opencv2/ml.hpp>
#include <iostream>
#include <fstream>
#include <vector>

using namespace cv;
using namespace std;

vector<vector<unsigned char>> readImage(const string &fileName) {
    ifstream file(fileName, ios::binary);

    //Initializing Variables
    char magicNumber[4];
    char sumOfImages[4];
    char sumOfRows[4];
    char sumOfCols[4];

    //Extracting data from file to aforementioned variables
    file.read(magicNumber, 4);
    file.read(sumOfImages, 4);
    file.read(sumOfRows, 4);
    file.read(sumOfCols, 4);


    //Big Endian Conversion
    int realNumberOfImages =
            (static_cast<unsigned char>(sumOfImages[0]) << 24) | (static_cast<unsigned char>(sumOfImages[1]) << 16) |
            (static_cast<unsigned char>(sumOfImages[2]) << 8) | static_cast<unsigned char>(sumOfImages[3]);
    int realNumberOfRows =
            (static_cast<unsigned char>(sumOfImages[0]) << 24) | (static_cast<unsigned char>(sumOfRows[1]) << 16) |
            (static_cast<unsigned char>(sumOfRows[2]) << 8) | static_cast<unsigned char>(sumOfRows[3]);
    int realNumberOfCols =
            (static_cast<unsigned char>(sumOfImages[0]) << 24) | (static_cast<unsigned char>(sumOfCols[1]) << 16) |
            (static_cast<unsigned char>(sumOfCols[2]) << 8) | static_cast<unsigned char>(sumOfCols[3]);

    vector<vector<unsigned char>> result;

    //Inserting the extracted data into a vector to be returned to the main program
    for (int i = 0; i < realNumberOfImages; i++) {
        vector<unsigned char> image(realNumberOfRows * realNumberOfCols);
        file.read((char *) (image.data()), realNumberOfRows * realNumberOfCols);
        result.push_back(image);
    }

    file.close();
    return result;
}

vector<vector<unsigned char>> readLabelFile(const string &filename) {
    ifstream file(filename, ios::binary);

    if (!file) {
        cerr << "Failed to open the IDX3-UBYTE file." << endl;
        return {};
    }

    //Initializing Variables
    char magicNumber[4];
    char numImagesBytes[4];

    //Reading files and putting the values into aforementioned variables
    file.read(magicNumber, 4);
    file.read(numImagesBytes, 4);

    //Big Endian Conversion
    int numImages = (static_cast<unsigned char>(numImagesBytes[0]) << 24) |
                    (static_cast<unsigned char>(numImagesBytes[1]) << 16) |
                    (static_cast<unsigned char>(numImagesBytes[2]) << 8) |
                    static_cast<unsigned char>(numImagesBytes[3]);

    // Initialize a vector to store the labels
    vector<vector<unsigned char>> result;

    // Putting the labels into vector to be returned to the main program
    for (int i = 0; i < numImages; i++) {
        std::vector<unsigned char> image(1);
        file.read((char *) (image.data()), 1);

        result.push_back(image);
    }

    file.close();
    return result;
}

int main() {
    //Read Binary Data from MNIST Database
    vector<vector<unsigned char>> pictures = readImage(
            "D:\\University\\CUHKSZ\\Courses\\Y2T1\\CSC3002\\MNIST Database\\train-images-idx3-ubyte\\train-images.idx3-ubyte");
    vector<vector<unsigned char>> labels = readLabelFile(
            "D:\\University\\CUHKSZ\\Courses\\Y2T1\\CSC3002\\MNIST Database\\train-labels-idx1-ubyte\\train-labels.idx1-ubyte");
    vector<vector<unsigned char>> testData = readImage("D:\\University\\CUHKSZ\\Courses\\Y2T1\\CSC3002\\MNIST Database\\t10k-images-idx3-ubyte\\t10k-images.idx3-ubyte");
    vector<vector<unsigned char>> testLabel = readLabelFile("D:\\University\\CUHKSZ\\Courses\\Y2T1\\CSC3002\\MNIST Database\\t10k-labels-idx1-ubyte\\t10k-labels.idx1-ubyte");

    // Initializing matrices for training and testing model
    Mat trainingImages = Mat::zeros(pictures.size(), 784, CV_32F);
    Mat trainingLabels = Mat::zeros(pictures.size(), 1, CV_32F);
    Mat testingImages = Mat::zeros(testData.size(), 784, CV_32F);
    Mat testingLabels = Mat::zeros(testData.size(), 1, CV_32F);

    // Inserting the data from vector into matrix for training process
    for (int i = 0; i < (int) pictures.size(); i++) {
        for (int j = 0; j < (int) pictures[i].size(); j++) {
            trainingImages.at<float>(i, j) =  pictures[i][j];
        }
        trainingLabels.at<float>(i, 0) = labels[i][0];
    }

    // Inserting the data from vector into matrix for testing process
    for (int i = 0; i < (int) testData.size(); i++) {
        for (int j = 0; j < (int) testData[i].size(); j++) {
            testingImages.at<float>(i, j) = testData[i][j];
        }
        testingLabels.at<float>(i, 0) = testLabel[i][0];
    }

    // Creating the ML model (K-nearest Neighbor)
    cv::Ptr<cv::ml::KNearest> knn = cv::ml::KNearest::create();

    // Start the training process
    cout << "Start Training" << endl;
    knn->train(trainingImages, ml::ROW_SAMPLE, trainingLabels);
    cout << "Finished Training" << endl;

    // Initialize variables to record testing accuracy
    int trial = 0;
    int success = 0;

    // Initializing a matrix to store predictions
    Mat expected = Mat::zeros(testLabel.size(), 1, CV_32F);

    // Use the ML model to predict test images
    knn->findNearest(testingImages, 1, expected);

    // Record the testing process and calculate the model's accuracy
    for(int i = 0; i < (int)testLabel.size(); i++){
        trial++;
        if((int) expected.at<float>(i, 0) == (int) testLabel[i][0]){
            success++;
        }
    }
    cout << "Accuracy: " << success << "/" << trial << " = " << ((float) success/ (float) trial) * 100 << "%" << endl;

    return 0;
}
