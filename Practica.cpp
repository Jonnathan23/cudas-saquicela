#include <iostream>
#include <filesystem>
#include <opencv2/opencv.hpp>
#include <opencv2/cudaarithm.hpp>
#include <opencv2/cudafilters.hpp>
#include <opencv2/cudaimgproc.hpp>
#include <opencv2/core/cuda.hpp>
#include <iomanip>

using namespace std;
namespace fs = std::filesystem;
using namespace cv;

// ────────────── UTILITARIOS ──────────────

string matTypeToStr(int type) {
    string r;
    uchar depth = type & CV_MAT_DEPTH_MASK;
    uchar chans = 1 + (type >> CV_CN_SHIFT);

    switch (depth) {
        case CV_8U:  r = "8U"; break;
        case CV_8S:  r = "8S"; break;
        case CV_16U: r = "16U"; break;
        case CV_16S: r = "16S"; break;
        case CV_32S: r = "32S"; break;
        case CV_32F: r = "32F"; break;
        case CV_64F: r = "64F"; break;
        default:     r = "User"; break;
    }

    r += "C" + to_string(chans);
    return r;
}

void logMatInfo(const string& label, const Mat& m) {
    double mem_mb = (m.total() * m.elemSize()) / (1024.0 * 1024.0);
    cout << label << " - Size: " << m.size() << ", Type: " << matTypeToStr(m.type()) 
         << ", Memory: " << fixed << setprecision(2) << mem_mb << " MB" << endl;
}

void printSeparator(const string& title) {
    cout << "\n" << string(60, '=') << endl;
    cout << "  " << title << endl;
    cout << string(60, '=') << endl;
}

void printStepHeader(const string& step, int stepNum, int totalSteps) {
    cout << "\n+-- PASO " << stepNum << "/" << totalSteps << ": " << step << " " << string(40 - step.length(), '-') << "+" << endl;
}

void printStepFooter(double time_ms, const string& details = "") {
    cout << "+-- Tiempo: " << fixed << setprecision(3) << time_ms << " ms";
    if (!details.empty()) {
        cout << " | " << details;
    }
    cout << " " << string(20, '-') << "+" << endl;
}

// ────────────── CPU ──────────────

void preprocessCPU(const Mat& frame,
                  Mat& out_blur,
                  Mat& out_erode,
                  Mat& out_dilate,
                  Mat& out_edges,
                  Mat& out_hist_eq,
                  double& elapsed_ms)
{
    printSeparator("PROCESAMIENTO CPU");
    cout << "🖥️  Procesando en CPU..." << endl;
    logMatInfo("Imagen original", frame);
    
    TickMeter tm_total; tm_total.start();
    TickMeter tm_step;
    
    // PASO 1: Gaussiano
    printStepHeader("Filtro Gaussiano", 1, 5);
    tm_step.start();
    GaussianBlur(frame, out_blur, Size(5,5), 1.5);
    tm_step.stop();
    logMatInfo("Resultado Gaussiano", out_blur);
    printStepFooter(tm_step.getTimeMilli(), "Kernel 5x5, sigma=1.5");

    // PASO 2: Morfología
    printStepHeader("Operaciones Morfológicas", 2, 5);
    tm_step.reset(); tm_step.start();
    Mat k = getStructuringElement(MORPH_RECT, Size(3,3));
    
    TickMeter tm_erode; tm_erode.start();
    erode(out_blur, out_erode, k);
    tm_erode.stop();
    
    TickMeter tm_dilate; tm_dilate.start();
    dilate(out_erode, out_dilate, k);
    tm_dilate.stop();
    
    tm_step.stop();
    logMatInfo("Resultado Erosión", out_erode);
    logMatInfo("Resultado Dilatación", out_dilate);
    cout << "  +-- Erosion: " << tm_erode.getTimeMilli() << " ms | Dilatacion: " << tm_dilate.getTimeMilli() << " ms" << endl;
    printStepFooter(tm_step.getTimeMilli(), "Kernel 3x3 rectangular");

    // PASO 3: Conversión a escala de grises
    printStepHeader("Conversión BGR → Gris", 3, 5);
    tm_step.reset(); tm_step.start();
    Mat gray;
    cvtColor(out_dilate, gray, COLOR_BGR2GRAY);
    tm_step.stop();
    logMatInfo("Imagen en escala de grises", gray);
    printStepFooter(tm_step.getTimeMilli());

    // PASO 4: Detección de bordes
    printStepHeader("Detección de Bordes (Canny)", 4, 5);
    tm_step.reset(); tm_step.start();
    Canny(gray, out_edges, 50, 150);
    tm_step.stop();
    logMatInfo("Bordes detectados", out_edges);
    printStepFooter(tm_step.getTimeMilli(), "Umbral bajo: 50, alto: 150");

    // PASO 5: Ecualización de histograma
    printStepHeader("Ecualización de Histograma", 5, 5);
    tm_step.reset(); tm_step.start();
    Mat ycrcb;
    TickMeter tm_convert1; tm_convert1.start();
    cvtColor(frame, ycrcb, COLOR_BGR2YCrCb);
    tm_convert1.stop();
    
    TickMeter tm_split; tm_split.start();
    vector<Mat> ch;
    split(ycrcb, ch);
    tm_split.stop();
    
    TickMeter tm_equalize; tm_equalize.start();
    equalizeHist(ch[0], ch[0]);
    tm_equalize.stop();
    
    TickMeter tm_merge; tm_merge.start();
    merge(ch, ycrcb);
    tm_merge.stop();
    
    TickMeter tm_convert2; tm_convert2.start();
    cvtColor(ycrcb, out_hist_eq, COLOR_YCrCb2BGR);
    tm_convert2.stop();
    tm_step.stop();
    
    logMatInfo("Imagen ecualizada", out_hist_eq);
    cout << "  +-- BGR->YCrCb: " << tm_convert1.getTimeMilli() << " ms | Split: " << tm_split.getTimeMilli() 
         << " ms | EqualizeHist: " << tm_equalize.getTimeMilli() << " ms" << endl;
    cout << "     Merge: " << tm_merge.getTimeMilli() << " ms | YCrCb->BGR: " << tm_convert2.getTimeMilli() << " ms" << endl;
    printStepFooter(tm_step.getTimeMilli(), "Canal Y ecualizado");

    tm_total.stop();
    elapsed_ms = tm_total.getTimeMilli();
    
    cout << "\n🏁 TIEMPO TOTAL CPU: " << fixed << setprecision(3) << elapsed_ms << " ms" << endl;
}

// ────────────── GPU ──────────────

void preprocessGPU(const Mat& frame,
                  Mat& out_blur,
                  Mat& out_erode,
                  Mat& out_dilate,
                  Mat& out_edges,
                  Mat& out_hist_eq,
                  double& elapsed_ms)
{
    printSeparator("PROCESAMIENTO GPU");
    cout << "🚀 Procesando en GPU..." << endl;
    
    TickMeter tm_total; tm_total.start();
    TickMeter tm_step;

    // PASO 0: Subir imagen a GPU
    printStepHeader("Transferencia CPU → GPU", 0, 6);
    tm_step.start();
    cuda::GpuMat d_frame(frame);
    tm_step.stop();
    cout << "Imagen subida a GPU - Size: " << d_frame.size() << ", Type: " << matTypeToStr(d_frame.type()) << endl;
    double upload_time = tm_step.getTimeMilli();
    printStepFooter(upload_time, "Upload de " + to_string((frame.total() * frame.elemSize()) / (1024.0 * 1024.0)) + " MB");

    // PASO 1: Filtro Gaussiano
    printStepHeader("Filtro Gaussiano GPU", 1, 6);
    cuda::GpuMat d_blur;
    tm_step.reset(); tm_step.start();
    auto gauss = cuda::createGaussianFilter(d_frame.type(), d_frame.type(), Size(5,5), 1.5);
    gauss->apply(d_frame, d_blur);
    tm_step.stop();
    printStepFooter(tm_step.getTimeMilli(), "Kernel 5x5, sigma=1.5");

    // PASO 2: Operaciones Morfológicas
    printStepHeader("Operaciones Morfológicas GPU", 2, 6);
    cuda::GpuMat d_erode, d_dilate;
    Mat k = getStructuringElement(MORPH_RECT, Size(3,3));
    tm_step.reset(); tm_step.start();
    
    // Conversión a escala de grises primero
    cuda::GpuMat d_blur_gray;
    TickMeter tm_gray; tm_gray.start();
    cuda::cvtColor(d_blur, d_blur_gray, COLOR_BGR2GRAY);
    tm_gray.stop();

    // Morfología
    TickMeter tm_morph; tm_morph.start();
    auto erF = cuda::createMorphologyFilter(MORPH_ERODE, d_blur_gray.type(), k);
    erF->apply(d_blur_gray, d_erode);
    auto dilF = cuda::createMorphologyFilter(MORPH_DILATE, d_erode.type(), k);
    dilF->apply(d_erode, d_dilate);
    tm_morph.stop();
    
    tm_step.stop();
    cout << "  +-- BGR->Gris: " << tm_gray.getTimeMilli() << " ms | Morfologia: " << tm_morph.getTimeMilli() << " ms" << endl;
    printStepFooter(tm_step.getTimeMilli(), "Kernel 3x3 rectangular");

    // PASO 3: Detección de bordes
    printStepHeader("Detección de Bordes GPU (Canny)", 3, 6);
    cuda::GpuMat d_edges;
    tm_step.reset(); tm_step.start();
    auto canny = cuda::createCannyEdgeDetector(50.0, 150.0);
    canny->detect(d_dilate, d_edges);
    tm_step.stop();
    printStepFooter(tm_step.getTimeMilli(), "Umbral bajo: 50, alto: 150");

    // PASO 4: Ecualización de histograma
    printStepHeader("Ecualización de Histograma GPU", 4, 6);
    cuda::GpuMat d_hist_eq;
    tm_step.reset(); tm_step.start();
    
    cuda::GpuMat d_ycrcb;
    TickMeter tm_convert1; tm_convert1.start();
    cuda::cvtColor(d_frame, d_ycrcb, COLOR_BGR2YCrCb);
    tm_convert1.stop();
    
    TickMeter tm_split; tm_split.start();
    vector<cuda::GpuMat> ch;
    cuda::split(d_ycrcb, ch);
    tm_split.stop();
    
    TickMeter tm_equalize; tm_equalize.start();
    cuda::equalizeHist(ch[0], ch[0]);
    tm_equalize.stop();
    
    TickMeter tm_merge; tm_merge.start();
    cuda::merge(ch, d_ycrcb);
    tm_merge.stop();
    
    TickMeter tm_convert2; tm_convert2.start();
    cuda::cvtColor(d_ycrcb, d_hist_eq, COLOR_YCrCb2BGR);
    tm_convert2.stop();
    
    tm_step.stop();
    cout << "  +-- BGR->YCrCb: " << tm_convert1.getTimeMilli() << " ms | Split: " << tm_split.getTimeMilli() 
         << " ms | EqualizeHist: " << tm_equalize.getTimeMilli() << " ms" << endl;
    cout << "     Merge: " << tm_merge.getTimeMilli() << " ms | YCrCb->BGR: " << tm_convert2.getTimeMilli() << " ms" << endl;
    printStepFooter(tm_step.getTimeMilli(), "Canal Y ecualizado");

    // PASO 5: Transferencia GPU → CPU
    printStepHeader("Transferencia GPU → CPU", 5, 6);
    tm_step.reset(); tm_step.start();
    d_blur.download(out_blur);
    d_erode.download(out_erode);
    d_dilate.download(out_dilate);
    d_edges.download(out_edges);
    d_hist_eq.download(out_hist_eq);
    tm_step.stop();
    double download_time = tm_step.getTimeMilli();
    
    // Información de las imágenes descargadas
    logMatInfo("GPU → CPU: Gaussiano", out_blur);
    logMatInfo("GPU → CPU: Erosión", out_erode);
    logMatInfo("GPU → CPU: Dilatación", out_dilate);
    logMatInfo("GPU → CPU: Canny", out_edges);
    logMatInfo("GPU → CPU: HistEq", out_hist_eq);
    
    double total_mb = ((out_blur.total() * out_blur.elemSize()) + 
                      (out_erode.total() * out_erode.elemSize()) +
                      (out_dilate.total() * out_dilate.elemSize()) + 
                      (out_edges.total() * out_edges.elemSize()) + 
                      (out_hist_eq.total() * out_hist_eq.elemSize())) / (1024.0 * 1024.0);
    printStepFooter(download_time, "Download de " + to_string(total_mb) + " MB");

    tm_total.stop();
    elapsed_ms = tm_total.getTimeMilli();
    
    cout << "\n🏁 TIEMPO TOTAL GPU: " << fixed << setprecision(3) << elapsed_ms << " ms" << endl;
    cout << "   +-- Transferencias (Upload + Download): " << (upload_time + download_time) << " ms (" 
         << fixed << setprecision(1) << ((upload_time + download_time) / elapsed_ms * 100) << "% del total)" << endl;
}

// ────────────── MAIN ──────────────

int main(int argc, char* argv[]) {
    printSeparator("BENCHMARK CPU vs GPU - OpenCV");
    
    const string ruta = "/home/bryam/Descargas/Ejemplo-17-Super-Resolution-DNN-OpenCV/San-Basilio.jpg";
    Mat frame = imread(ruta, IMREAD_COLOR);
    if(frame.empty()) {
        cerr << "❌ No se pudo cargar imagen en: " << ruta << endl;
        return -1;
    }

    cout << "📁 Imagen cargada: " << ruta << endl;
    logMatInfo("Imagen original", frame);

    // Información del sistema
    cout << "\n💻 INFORMACIÓN DEL SISTEMA:" << endl;
    cout << "   OpenCV Version: " << CV_VERSION << endl;
    cout << "   Threads disponibles: " << getNumThreads() << endl;
    
    // Info CUDA
    if (cuda::getCudaEnabledDeviceCount() == 0) {
        cerr << "❌ No hay dispositivos CUDA disponibles." << endl;
        return -1;
    }
    
    cout << "\n🔥 INFORMACIÓN CUDA:" << endl;
    cout << "   Dispositivos CUDA: " << cuda::getCudaEnabledDeviceCount() << endl;
    cuda::printShortCudaDeviceInfo(cuda::getDevice());
    
    // Información básica de GPU (sin cudaMemGetInfo)
    cout << "   GPU Device ID: " << cuda::getDevice() << endl;

    // Contenedores
    Mat cpu_blur, cpu_erode, cpu_dilate, cpu_edges, cpu_hist;
    Mat gpu_blur, gpu_erode, gpu_dilate, gpu_edges, gpu_hist;
    double time_cpu = 0, time_gpu = 0;

    // Procesar
    preprocessCPU(frame, cpu_blur, cpu_erode, cpu_dilate, cpu_edges, cpu_hist, time_cpu);
    preprocessGPU(frame, gpu_blur, gpu_erode, gpu_dilate, gpu_edges, gpu_hist, time_gpu);

    // RESULTADOS FINALES
    printSeparator("RESULTADOS COMPARATIVOS");
    cout << "📊 TIEMPOS DE EJECUCIÓN:" << endl;
    cout << "   🖥️  CPU: " << fixed << setprecision(3) << time_cpu << " ms" << endl;
    cout << "   🚀 GPU: " << fixed << setprecision(3) << time_gpu << " ms" << endl;
    cout << "   📈 Speedup: " << fixed << setprecision(2) << (time_cpu / time_gpu) << "x ";
    
    if (time_gpu < time_cpu) {
        cout << "(GPU es " << fixed << setprecision(1) << ((time_cpu - time_gpu) / time_cpu * 100) << "% más rápido)" << endl;
    } else {
        cout << "(CPU es " << fixed << setprecision(1) << ((time_gpu - time_cpu) / time_gpu * 100) << "% más rápido)" << endl;
    }
    
    // Estadísticas adicionales
    double fps_cpu = 1000.0 / time_cpu;
    double fps_gpu = 1000.0 / time_gpu;
    cout << "   🎥 FPS equivalente CPU: " << fixed << setprecision(1) << fps_cpu << " fps" << endl;
    cout << "   🎥 FPS equivalente GPU: " << fixed << setprecision(1) << fps_gpu << " fps" << endl;

    // Mostrar imágenes
    cout << "\n🖼️  Mostrando resultados visuales..." << endl;
    imshow("Original", frame);
    imshow("CPU - Gaussiano", cpu_blur);
    imshow("GPU - Gaussiano", gpu_blur);
    imshow("CPU - Erosion", cpu_erode);
    imshow("GPU - Erosion", gpu_erode);
    imshow("CPU - Dilatacion", cpu_dilate);
    imshow("GPU - Dilatacion", gpu_dilate);
    imshow("CPU - Canny", cpu_edges);
    imshow("GPU - Canny", gpu_edges);
    imshow("CPU - HistEq", cpu_hist);
    imshow("GPU - HistEq", gpu_hist);

    cout << "\n⌨️  Presiona cualquier tecla para salir..." << endl;
    waitKey(0);
    return 0;
}