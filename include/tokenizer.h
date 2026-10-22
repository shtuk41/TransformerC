#pragma once

#include <algorithm> 
#include <cstdint> 
#include <limits>
#include <string>
#include <vector>

#include <utf8.h>

//#define TOKENIZER_DEBUG

//BYTE PAIR ENCODING
class BPE_Tokenizer
{
private:
	std::map<uint32_t, std::string> vocabluary;
	uint32_t maxTokenNumber = 0;
	std::vector<uint32_t> code_points;

	std::string uint32ToString(uint32_t c)
	{
		std::string utf8charrep;
		utf8::append(c, std::back_inserter(utf8charrep));
		return utf8charrep;
		//return std::string(1, static_cast<char>(c));
	}

public:
	BPE_Tokenizer(std::vector<uint32_t> &code_points) : 
											maxTokenNumber(0),
											code_points(code_points)
	{
		for (int ii = 0; ii < code_points.size(); ii++)
		{
			if (code_points[ii] > maxTokenNumber)
			{
				maxTokenNumber = code_points[ii];
			}

			if (!vocabluary.contains(code_points[ii]))
			{
				auto rep = uint32ToString(code_points[ii]);
				vocabluary[code_points[ii]] = rep;
			}
		}
	}

	auto GetVocabluary() const { return vocabluary; }
	uint32_t GetMaxTokenNumber() const { return maxTokenNumber;  }

	size_t GetVectorSize() const { return code_points.size(); }
	
	auto GetCodePoints() const 
	{
		return code_points;
	}
	
	std::string Decode(std::vector<uint32_t> &codes)
	{
		std::string utf8charrep;
		
		utf8charrep.reserve(codes.size() * 8); 
		
		for (auto &c : codes)
		{
			utf8charrep += vocabluary[c];
		}	
		
		return utf8charrep;
	}

	void TrainStep()
	{
		std::map<std::pair<uint32_t, uint32_t>, size_t> pairs;

		for (int ii = 0; ii < code_points.size() - 1; ii += 1)
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

#ifdef TOKENIZER_DEBUG
		std::cout << std::format("pair ({},{}), times {}\n", max_pair.first, max_pair.second, max);
		std::cout << std::format("total pairs: {}\n", pairs.size());
		std::cout << std::format("code point vector size: {}\n", code_points.size());
#endif
		
		maxTokenNumber += 1;
		//dictionary[max_pair] = maxTokenNumber;

		vocabluary[maxTokenNumber] = vocabluary[max_pair.first] + vocabluary[max_pair.second];

		for (size_t ii = 0; ii + 1 < code_points.size(); )
		{
			if (code_points[ii] == max_pair.first && code_points[ii + 1] == max_pair.second)
			{
				code_points[ii] = maxTokenNumber; 
				code_points[ii + 1] = std::numeric_limits<std::uint32_t>::max(); 

				ii += 2; // Jump by 2 to completely skip this pair
			}
			else
			{
				ii += 1; // Slide forward by 1 to check overlapping pairs
			}
		}

		code_points.erase(
			std::remove(code_points.begin(), code_points.end(), std::numeric_limits<std::uint32_t>::max()),
			code_points.end()
		);
	}
};
