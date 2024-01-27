/* Demo application for Computer Vision Library.
 * @file
 * @date 2018-09-05
 * @author Anonymous
 */

#include <cvlib.hpp>
#include <opencv2/opencv.hpp>

int demo_motion_segmentation(int argc, char* argv[])
{
    cv::VideoCapture cap(0, 200);
    if (!cap.isOpened())
        return -1;
    

    auto mseg = cvlib::motion_segmentation(); // \todo use cvlib::motion_segmentation
    const auto main_wnd = "orig";
    const auto demo_wnd = "demo";

    int threshold = 200;
    cv::namedWindow(main_wnd);
    cv::namedWindow(demo_wnd);
    cv::createTrackbar("th", demo_wnd, &threshold, 1000);

    cv::Mat frame;
    cv::Mat frame_mseg;
    while (cv::waitKey(30) != 27) // ESC
    {
        cap >> frame;
        cv::imshow(main_wnd, frame);
        cv::cvtColor(frame, frame, cv::COLOR_BGR2GRAY);
        frame.convertTo(frame, CV_32FC1);
        
        mseg.TackbarCallback(threshold); // \todo use TackbarCallback
        mseg.apply(frame, frame_mseg);
        mseg.getBackgroundImage(frame);
        if (!frame_mseg.empty())
            cv::imshow(demo_wnd, frame_mseg);
    }

    cv::destroyWindow(main_wnd);
    cv::destroyWindow(demo_wnd);

    return 0;
}
