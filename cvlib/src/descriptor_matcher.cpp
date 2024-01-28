/* Descriptor matcher algorithm implementation.
 * @file
 * @date 2018-11-25
 * @author Anonymous
 */

#include "cvlib.hpp"

namespace cvlib
{
void descriptor_matcher::knnMatchImpl(cv::InputArray queryDescriptors, std::vector<std::vector<cv::DMatch>>& matches, int k /*unhandled*/,
                                      cv::InputArrayOfArrays masks /*unhandled*/, bool compactResult /*unhandled*/)
{
    if (trainDescCollection.empty())
        return;

    auto q_desc = queryDescriptors.getMat();
    auto& t_desc = trainDescCollection[0];
    auto size = 32;

    matches.clear();
    matches.resize(q_desc.rows);

    for (int i = 0; i < q_desc.rows; ++i)
    {
        int min_distance = ratio_;
        int t_row_min = 0;
        bool find_flag = false;
        for (int j = 0; j < t_desc.rows; ++j)
        {
            auto q_desc_ptr = q_desc.ptr(i, 0);
            auto t_desc_ptr = t_desc.ptr(j, 0);
            
        	int distance = 0;
    		uint8_t tmp;

    		for (auto i = 0; i < size; i++)
    		{
        		tmp = *(q_desc_ptr + i) ^ *(t_desc_ptr + i);
        		while (tmp)
        		{
           	 	++distance;
            		tmp &= tmp - 1;
        		}
    		}
            
        	if (distance < min_distance)
        	{
            	min_distance = distance;
            	t_row_min = j;
            	find_flag = true;
        	}
        }
        if (find_flag)
            matches[i].emplace_back(i, t_row_min, min_distance);
    }
}

void descriptor_matcher::radiusMatchImpl(cv::InputArray queryDescriptors, std::vector<std::vector<cv::DMatch>>& matches, float /*maxDistance*/,
                                         cv::InputArrayOfArrays masks /*unhandled*/, bool compactResult /*unhandled*/)
{
    // \todo implement matching with "maxDistance"
    knnMatchImpl(queryDescriptors, matches, 1, masks, compactResult);
}
} // namespace cvlib
