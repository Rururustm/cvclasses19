/* FAST corner detector algorithm implementation.
 * @file
 * @date 2018-10-16
 * @author Anonymous
 */

#include "cvlib.hpp"

#include <ctime>

namespace cvlib
{
cv::Ptr<corner_detector_fast> corner_detector_fast::create()
{
    return cv::makePtr<corner_detector_fast>();
}

bool isKeyPoint(const cv::Mat&& m) {
    unsigned char threshold = 7;
    cv::Point center(m.cols / 2, m.rows / 2);
    using Point = cv::Point;
    std::array<cv::Point, 4> fst_circle = {Point(center.x + m.cols / 2, center.y), Point(center.x, center.y + m.rows / 2),
                                           Point(center.x - m.cols / 2, center.y), Point(center.x, center.y - m.rows / 2)};

    int upper = 0;
    int lower = 0;
    unsigned char i = std::min(static_cast<int>(m.at<unsigned char>(m.cols / 2, m.rows / 2)) + threshold, 255);
	unsigned char j = std::max(static_cast<int>(m.at<unsigned char>(m.cols / 2, m.rows / 2)) - threshold, 0);
    for (size_t ind = 0; ind < fst_circle.size(); ++ind) {
        if (i < m.at<unsigned char>(fst_circle[ind].x, fst_circle[ind].y))  ++upper; 
        else if (j > m.at<unsigned char>(fst_circle[ind].x, fst_circle[ind].y)) ++lower;
    }
    
    if ((upper > 2 || lower > 2)) {
        std::array<cv::Point, 12> snd_circle = {Point(center.x + m.cols / 2, center.y + 1), Point(center.x + m.cols / 2, center.y - 1),
                                                Point(center.x - m.cols / 2, center.y + 1), Point(center.x - m.cols / 2, center.y - 1),
                                                Point(center.x + 1, center.y + m.rows / 2), Point(center.x - 1, center.y + m.rows / 2),
                                                Point(center.x + 1, center.y - m.rows / 2), Point(center.x - 1, center.y - m.rows / 2),
                                                Point(center.x + m.cols / 2, center.y + m.rows / 2), Point(center.x - m.cols / 2, center.y - m.rows / 2),
                                                Point(center.x - m.cols / 2, center.y + m.rows / 2), Point(center.x + m.cols / 2, center.y - m.rows / 2)};
        
        for (size_t ind = 0; ind < snd_circle.size(); ++ind) {
                if (i < m.at<unsigned char>(snd_circle[ind].x, snd_circle[ind].y)) ++upper;
                else if (j > m.at<unsigned char>(snd_circle[ind].x, snd_circle[ind].y)) ++lower;
        }
        if (upper >= 16 || lower >= 16) return true;
    }

    return false;
}

void corner_detector_fast::detect(cv::InputArray image, CV_OUT std::vector<cv::KeyPoint>& keypoints, cv::InputArray /*mask = cv::noArray()*/)
{
    keypoints.clear();
    if(image.empty()) return;
    cv::Mat image_mat;
	image.getMat().copyTo(image_mat);
    if (image_mat.channels() == 3)
	    cv::cvtColor(image_mat, image_mat, cv::COLOR_BGR2GRAY);
	int border = 3;
	cv::Mat border_image(image_mat.rows + border * 2, image_mat.cols + border * 2, image_mat.depth());
	cv::copyMakeBorder(image_mat, border_image, border, border, border, border, cv::BORDER_CONSTANT);

	for (int i = border; i < border_image.rows - border; ++i)
		for (int j = border; j < border_image.cols - border; j++)
			if (isKeyPoint(border_image(cv::Range(i - border, i + border + 1), cv::Range(j - border, j + border + 1)))) 
            keypoints.push_back(cv::KeyPoint(j, i, 2*border + 1));
}

void corner_detector_fast::compute(cv::InputArray, std::vector<cv::KeyPoint>& keypoints, cv::OutputArray descriptors)
{
    const int desc_length = 2;
    descriptors.create(static_cast<int>(keypoints.size()), desc_length, CV_32S);
    auto desc_mat = descriptors.getMat();
    desc_mat.setTo(0);

    int* ptr = reinterpret_cast<int*>(desc_mat.ptr());
    for (const auto& pt : keypoints)
    {
        for (int i = 0; i < desc_length; ++i)
        {
            *ptr = std::rand();
            ++ptr;
        }
    }
}

void corner_detector_fast::detectAndCompute(cv::InputArray, cv::InputArray, std::vector<cv::KeyPoint>&, cv::OutputArray descriptors, bool /*= false*/)
{

}
} // namespace cvlib
