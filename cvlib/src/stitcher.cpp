#include <cmath>
#include <vector>

#include "cvlib.hpp"

namespace cvlib
{
Stitcher::Stitcher()
{
    this->detector = cv::ORB::create();
    this->matcher = cv::BFMatcher::create();
}

void Stitcher::initialize(cv::InputArray input)
{
    input.getMat().copyTo(this->dst);
}

void Stitcher::stitch(cv::InputArray input, cv::OutputArray output)
{
    cv::Mat src;
    input.getMat().copyTo(src);
    
    std::vector<cv::KeyPoint> src_corners;
    std::vector<cv::KeyPoint> dst_corners;
    
    cv::Mat src_descriptors;
    cv::Mat dst_descriptors;

    this->detector->detectAndCompute(src, cv::noArray(), src_corners, src_descriptors);
    this->detector->detectAndCompute(this->dst, cv::noArray(), dst_corners, dst_descriptors);

    std::vector<std::vector<cv::DMatch>> matches;
    this->matcher->radiusMatch(src_descriptors, dst_descriptors, matches, 100.0f);

    std::vector<cv::Point2f> src_keys, dst_keys;
    for (const auto& match : matches)
        if (!match.empty())
        {
            src_keys.push_back(src_corners[match[0].queryIdx].pt);
            dst_keys.push_back(dst_corners[match[0].queryIdx].pt);
        }
    if ((src_keys.size() < 4 || dst_keys.size() < 4))
        return;

    cv::Mat homography = cv::findHomography(cv::Mat(src_keys), cv::Mat(dst_keys), cv::RANSAC);

    const auto dst_size = cv::Size(this->dst.cols + src.cols, this->dst.rows);
    auto dst = cv::Mat(dst_size, CV_8U);
    cv::warpPerspective(src, dst, homography, dst.size(), cv::INTER_CUBIC);
    cv::Mat roi = cv::Mat(dst, cv::Rect(0, 0, dst.cols, dst.rows));

    dst.copyTo(output);
}
} // namespace cvlib
