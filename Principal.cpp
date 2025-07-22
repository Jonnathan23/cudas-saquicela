

#include <chrono>
#include <cmath>
#include <cstdlib>
#include <cstring>
#include <fstream>
#include <iostream>
#include <random>
#include <sstream>

// Librerías de OpenCV
// #include <opencv2/opencv.hpp>

#include <opencv2/core.hpp>      // Librería que contiene las estructuras y funciones base para representar las imágenes como matrices
#include <opencv2/highgui.hpp>   // Librería para crear interfaces básicas de usuario
#include <opencv2/imgcodecs.hpp> // Librería contiene los métodos para cargar imágenes de acuerdo a un códec
#include <opencv2/imgproc.hpp>   // Librería que contiene operaciones para realizar procesamiento de imágenes
#include <opencv2/video.hpp>     // Librería para realizar lectura de vídeos
#include <opencv2/videoio.hpp>   // Librería para escritura de vídoeos y streaming
// #include <opencv2/xfeatures2d.hpp>

#include <opencv2/dnn_superres.hpp>
#include <opencv2/superres/optical_flow.hpp>

#include <opencv2/features2d/features2d.hpp>

#include <opencv2/dnn.hpp>

using namespace std;
using namespace cv;
using namespace dnn;
using namespace dnn_superres;

DnnSuperResImpl sr;

Mat superResolucion(Mat img) {
    Mat res;
    sr.upsample(img, res);

    return res;
}

int main(int argc, char *args[]) {
    string path = "./models/FSRCNN_x4.pb";
    string modelo = "fsrcnn";
    int escala = 4;

    sr.readModel(path);
    sr.setModel(modelo, escala);

    // sr.setPreferableBackend(dnn::DNN_BACKEND_CUDA);
    // sr.setPreferableTarget(dnn::DNN_TARGET_CUDA);

    VideoCapture video("Moscow Russia Aerial Drone 4K Timelab.pro.mp4");
    auto startTime = chrono::high_resolution_clock::now();
    int frameCount = 0;
    double fps = 0.0;

    if (video.isOpened()) {
        Mat image;
        Mat imgROI;
        string path = "./models/FSRCNN_x4.pb";
        string modelo = "fsrcnn";
        Mat resultado;
        Mat escalada;

        Rect roi;
        namedWindow("Image", WINDOW_AUTOSIZE);
        namedWindow("ROI", WINDOW_AUTOSIZE);
        namedWindow("Escalada", WINDOW_AUTOSIZE);
        namedWindow("SuperRes", WINDOW_AUTOSIZE);

        while (3 == 3) {
            video >> image;

            if (image.empty()) {
                break;
            }

            if (imgROI.empty()) {
                roi.x = (image.cols / 2) - ((int)(image.cols / 2) * 0.3);
                roi.y = (image.rows / 2) - ((int)(image.rows / 2) * 0.3);
                roi.width = ((int)(image.cols / 2) * 0.3) * 2;
                roi.height = ((int)(image.rows / 2) * 0.3) * 2;
            }

            frameCount++;
            auto currentTime = chrono::high_resolution_clock::now();
            auto duration = chrono::duration_cast<chrono::milliseconds>(currentTime - startTime).count();

            if (duration > 1000) {
                fps = (double)frameCount / (duration / 1000.0);
                frameCount = 0;
                startTime = currentTime;
            }

            imgROI = image(roi);
            resultado = superResolucion(imgROI);
            resize(imgROI, escalada, Size(), escala, escala);

            cv::putText(resultado, "FPS: " + std::to_string(fps), cv::Point(10, 30), cv::FONT_HERSHEY_SIMPLEX, 1, cv::Scalar(0, 255, 0), 2);

            imshow("Image", image);
            imshow("ROI", imgROI);
            imshow("Escalada", escalada);
            imshow("SuperRes", resultado);

            if (waitKey(23) == 27) {
                break;
            }
        }
        video.release();
        destroyAllWindows();
    }
    return 0;
}