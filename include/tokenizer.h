#pragma once

#include <map>

//#define TOKENINZER_DEBUG

class BPE_Tokenizer
{
private:
	std::map<std::pair<uint32_t, uint32_t>, uint32_t> dictionary;
	uint32_t maxTokenNumber = 0;
	std::vector<uint32_t> code_points;

public:
	BPE_Tokenizer(std::vector<uint32_t> &code_points) : 
											maxTokenNumber(0),
											code_points(code_points)
	{
		for (int ii = 0; ii < code_points.size(); ii++)
		{
			if (code_points[ii] > maxTokenNumber)
				maxTokenNumber = code_points[ii];
		}
	}

	uint32_t GetMaxTokenNumber() const { return maxTokenNumber;  }

	size_t GetVectorSize() const { return code_points.size(); }

	void TrainStep()
	{
		std::map<std::pair<uint32_t, uint32_t>, size_t> pairs;

		for (int ii = 0; ii < code_points.size() - 1; ii += 2)
		{
			std::pair<uint32_t, uint32_t> next = { code_points[ii], code_points[ii + 1] };
			pairs[next] += 1;
		}

		size_t max = 0;
		std::pair<uint32_t, uint32_t> max_pair;

		for (const auto& kv : pairs)
		{
			if (kv.second > max)
			{
				max = kv.second;
				max_pair = kv.first;
			}
		}

#ifdef TOKENINZER_DEBUG
		std::cout << std::format("pair ({},{}), times {}\n", max_pair.first, max_pair.second, max);
		std::cout << std::format("total pairs: {}\n", pairs.size());
		std::cout << std::format("code point vector size: {}\n", code_points.size());
#endif
		
		maxTokenNumber += 1;
		dictionary[max_pair] = maxTokenNumber;

		for (size_t ii = 0; ii + 1 < code_points.size(); )
		{
			if (code_points[ii] == max_pair.first && code_points[ii + 1] == max_pair.second)
			{
				code_points[ii] = maxTokenNumber; // Replace first element
				code_points[ii + 1] = -1; // Mark second element for removal

				ii += 2; // Jump by 2 to completely skip this pair
			}
			else
			{
				ii += 1; // Slide forward by 1 to check overlapping pairs
			}
		}

		code_points.erase(
			std::remove(code_points.begin(), code_points.end(), -1),
			code_points.end()
		);
	}



};