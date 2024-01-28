/* Demo application for Computer Vision Library.
 * @file
 * @date 2018-09-05
 * @author Anonymous
 */

#include <cvlib.hpp>
#include <opencv2/opencv.hpp>

#include "utils.hpp"

void hist(cv::Mat& input, cv::Mat& output)
{
    int hist_size = 256;
    float range[] = {0, 256};
    const float *hist_range = {range};
    cv::Mat hist;
    cv::calcHist(&input, 1, 0, cv::Mat(), hist, 1, &hist_size, &hist_range, true, false);
    normalize(hist, hist, 0, input.rows, cv::NORM_MINMAX, -1, cv::Mat());
    int padding = 10;
    int total_padding = padding * (256 - 1);
    int max_width = (800 - total_padding) / 256;
    cv::Mat haming_hist(500, 800, CV_8UC3, cv::Scalar(255, 255, 255));

    for (int i = 0; i < 256; i++)
    {
        rectangle(haming_hist,  cv::Point((max_width + padding) * i, 500),
                  cv::Point((max_width + padding) * i + max_width, 500 - cvRound(hist.at<float>(i * (hist_size / 256)))),
                  cv::Scalar(255, 0, 0), cv::FILLED);
    }

    haming_hist.copyTo(output);
}

int demo_feature_descriptor(int argc, char* argv[])
{
    cv::VideoCapture cap(0);
    if (!cap.isOpened())
        return -1;

    const auto main_wnd = "orig";
    const auto demo_wnd = "demo";

    cv::namedWindow(main_wnd);
    cv::namedWindow(demo_wnd);

    cv::Mat frame;
    auto detector_a = cvlib::corner_detector_fast::create();
    auto detector_b = cv::ORB::create();
    std::vector<cv::KeyPoint> corners;
    
    utils::fps_counter fps;
    int pressed_key = 0;
    while (pressed_key != 27) // ESC
    {
        cap >> frame;
        cv::imshow(main_wnd, frame);

        detector_a->detect(frame, corners);
        cv::drawKeypoints(frame, corners, frame, cv::Scalar(0, 0, 255));

        utils::put_fps_text(frame, fps);
        cv::putText(frame,"count of the detected corners:" + std::to_string(corners.size()), cv::Point2f(50,50), cv::FONT_HERSHEY_PLAIN, 2,cv::Scalar(0,255,0)); 
        cv::imshow(demo_wnd, frame);

        pressed_key = cv::waitKey(30);
        if (pressed_key == ' ') // space
        {
            const auto hist_wnd = "hist";
            cv::namedWindow(hist_wnd);

            cv::Mat descriptors_cvlib;
            cv::Mat descriptors_cv;
            
            detector_a->compute(frame, corners, descriptors_cvlib);
            detector_b->compute(frame, corners, descriptors_cv);
            
            cv::resize(descriptors_cv, descriptors_cv, descriptors_cvlib.size());
            

            cv::Mat hamming_dist = cv::Mat(descriptors_cvlib.size(), descriptors_cvlib.type());
            cv::bitwise_xor(descriptors_cv, descriptors_cvlib, hamming_dist);

            cv::Mat hamming_hist;
            hist(hamming_dist, hamming_hist);
            cv::imshow(hist_wnd, hamming_hist);

            std::cout << "Dump descriptors complete! \n";
        }

        std::cout << "Feature points: " << corners.size() << "\r";
    }

    cv::destroyWindow(main_wnd);
    cv::destroyWindow(demo_wnd);

    return 0;
}
