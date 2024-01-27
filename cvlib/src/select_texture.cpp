/* Split and merge segmentation algorithm implementation.
 * @file
 * @date 2018-09-18
 * @author Anonymous
 */

#include "cvlib.hpp"

namespace
{
struct descriptor : public std::vector<double>
{
    using std::vector<double>::vector;
    descriptor operator-(const descriptor& right) const
    {
        descriptor temp = *this;
        for (size_t i = 0; i < temp.size(); ++i)
        {
            temp[i] -= right[i];
        }
        return temp;
    }

    double norm_l1() const
    {
        double res = 0.0;
        for (auto v : *this)
        {
            res += std::abs(v);
        }
        return res;
    }

    double norm_l2() const
    {
        double res = 0.0;
        for (const auto v : *this)
        {
            res += v * v;
        }
        return std::sqrt(res);
    }
};

void calculateDescriptor(const cv::Mat& image, int kernel_size, descriptor& descr)
{
    descr.clear();
    const double th_ = CV_PI / 4;
    const double lm_ = 5.0;
    const double gm = 0.5;
    cv::Mat response;
    cv::Mat mean;
    cv::Mat dev;

    for (auto sig = 5; sig <= 15; sig += 5)
        for (auto th = th_; th < 2*th_; th += th_)
            for (auto lm  = lm_; lm < 2*lm_; lm += lm_)
                    {
                        cv::Mat kernel = cv::getGaborKernel(cv::Size(kernel_size, kernel_size), sig, th, lm, gm);
                        cv::filter2D(image, response, CV_32F, kernel);
                        cv::meanStdDev(response, mean, dev);
                        descr.emplace_back(mean.at<double>(0));
                        descr.emplace_back(dev.at<double>(0));
                    }
}
} // namespace

namespace cvlib
{
cv::Mat select_texture(const cv::Mat& image, const cv::Rect& roi, double eps)
{
    cv::Mat imROI = image(roi);

    const int kernel_size_ = std::min(roi.height, roi.width) / 2; // \todo round to nearest odd
    const int kernel_size = std::round( kernel_size_ * 0.5f ) * 2.0f;

    descriptor reference;
    calculateDescriptor(image(roi), kernel_size, reference);

    cv::Mat res = cv::Mat::zeros(image.size(), CV_8UC1);

    descriptor test(reference.size());
    cv::Rect baseROI = roi - roi.tl();

    // \todo move ROI smoothly pixel-by-pixel
    for (int i = 0; i < image.size().width - roi.width + 1; ++i)
    {
        for (int j = 0; j < image.size().height - roi.height + 1; ++j)
        {

            auto curROI = baseROI + cv::Point(i, j);
            calculateDescriptor(image(curROI), kernel_size, test);

            // \todo implement and use norm L2
            res(curROI) = 255 * ((test - reference).norm_l2() <= eps);
        }
    }

    return res;
}
} // namespace cvlib
