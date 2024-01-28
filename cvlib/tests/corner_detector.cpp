/* FAST corner detector algorithm testing.
 * @file
 * @date 2018-09-05
 * @author Anonymous
 */

#include <catch2/catch.hpp>

#include "cvlib.hpp"

using namespace cvlib;

TEST_CASE("simple check corner", "[corner_detector_fast]")
{
    auto fast = corner_detector_fast::create();
    cv::Mat image = cv::Mat(10, 10, CV_8UC1, cv::Scalar(100));
    SECTION("empty image")
    {
        std::vector<cv::KeyPoint> out;
        fast->detect(image, out);
        REQUIRE(out.empty());
    }

    SECTION("center")
    {
        cv::Mat image = (cv::Mat_<unsigned char>(1, 1) << 255);
        std::vector<cv::KeyPoint> v;
        fast->detect(image, v);
        REQUIRE(1 == v.size());
    }
}	
