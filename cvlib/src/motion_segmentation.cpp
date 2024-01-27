/* Split and merge segmentation algorithm implementation.
 * @file
 * @date 2018-09-18
 * @author Anonymous
 */

#include "cvlib.hpp"

#include <iostream>
#include <vector>
#include <algorithm>

namespace cvlib
{
void motion_segmentation::apply(cv::InputArray _image, cv::OutputArray _fgmask, double)
{
    // \todo implement your own algorithm:
    //       * MinMax
    //       * Mean
    //       * 1G
    //       * GMM
    cv::Mat image = _image.getMat();
    _fgmask.create( image.size(), CV_8UC1 );
    cv::Mat& fgmask = _fgmask.getMatRef();

    size_t frames_amount = 3;

    if (frame_counter < frames_amount) {
        frames.push_back(image);
        ++frame_counter;
        return;
    }
    else {
        frames.pop_front();
        frames.push_back(image);
    }

    cv::Mat d = frames[0];
    cv::Mat d_f, d_s;
    for (int i = 1; i < frames.size() - 1; ++i) {
        cv::absdiff(frames[i], frames[i - 1], d_f);
        cv::absdiff(frames[i], frames[i + 1], d_s);
        d = cv::max(d_f, d_s);
    }

    cv::Mat min = frames[0];
    for (int i = 1; i < frames.size(); ++i) {
        min = cv::min(min, frames[i]);
    }
    cv::Mat max = frames[0];
    for (int i = 1; i < frames.size(); ++i) {
        max = cv::max(max, frames[i]);
    }

    std::vector<double>vec;
    if (d.isContinuous())
    vec.assign(d.begin<double>(), d.end<double>());
    std::sort(vec.begin(), vec.end());

    for (int i = 0; i < fgmask.rows; ++i) {
        for (int j = 0; j < fgmask.cols; ++j) {
            if ((cv::abs(max.at<double>(i,j) - image.at<double>(i,j)) > (vec[vec.size() / 2] * threshold / 100)) || 
            (cv::abs(min.at<double>(i,j) - image.at<double>(i,j)) > (vec[vec.size() / 2] * threshold / 100))) {
                fgmask.at<short>(i,j) =  255;
            }
            else {
                fgmask.at<short>(i,j) = 0;
            }
        }
    }
}
} // namespace cvlib
