
#pragma once

#include <format>
#include <algorithm>
#include <fstream>
#include <iostream>
#include <set>
#include <stdexcept>
#include <string>
#include <vector>
#include <unordered_map>

#include <torch/torch.h>
#include <utf8.h>


struct encode
{
	std::unordered_map<char, int> characters;
	
	encode(const std::set<char> &s)
	{
		int index = 0;
		for (auto c : s)
		{
			characters[c] = index++;
		}
	}
	
	int operator()(char c) const
	{
		return characters.at(c);
	}
	
	std::vector<int> operator()(std::string s) const
	{
		std::vector<int> enc;
		enc.reserve(s.size());
		for (auto c : s)
		{
			enc.push_back((*this)(c));
		}
		
		return enc;
	}
	
	std::vector<int> operator()(std::vector<char> &s) const
	{
		std::vector<int> enc;
		enc.reserve(s.size());
		for (auto c : s)
		{
			enc.push_back((*this)(c));
		}
		
		return enc;
	}
};

template <class T = int64_t>
struct decode
{
	std::unordered_map<T, char> characters;
	
	decode(const std::set<char> &s)
	{
		T index = 0;
		for (auto c : s)
		{
			characters[index++] = c;
		}
	}
	
	char operator()(T index) const
	{
		return characters.at(index);
	}
	
	std::vector<char> operator()(std::vector<T> &encoded) const
	{
		std::vector<char> dec;
		dec.reserve(encoded.size());
		for (auto ii : encoded)
		{
			dec.push_back((*this)(ii));
		}
		
		return dec;
	}
	
	void print()
	{
		for (const auto&[t1, t2] : characters)
		{
			std::cout << t1 << " " << t2 << std::endl;
		}
	}
};

std::vector<char> readInputData()
{
	std::ifstream inputFile("input.txt", std::ios::binary);
	
	if (!inputFile.is_open())
	{
		std::cout << "Unable to open input.txt" << std::endl;
		throw std::runtime_error("unable to open input.txt"); 
	}
	
	inputFile.seekg(0, std::ios::end);
	std::streamsize size = inputFile.tellg();
	
	std::cout << std::format("input.txt size: {}\n", size);
	
	inputFile.seekg(0, std::ios::beg);
	
	std::vector<char> buffer(size);

	if (inputFile.read(buffer.data(), size))
	{
		return buffer;
	}
	
	throw std::runtime_error("Failed to read file input.txt");
}

void printCharacters(std::vector<char> &v, int start, int number)
{
	std::vector<char>::iterator it = v.begin() + start;
	
	for (; it != v.end(); ++it)
	{
		std::cout << *it;
		number--;
		
		if (number == 0)
			break;
	}
	
	std::cout << std::endl;
}

void printTensorShape(torch::Tensor &t)
{
	std::cout << "Shape: ";
	
	for (auto dim : t.sizes())
	{
		std::cout << dim << " ";
	}
	
	std::cout << std::endl;
}

std::vector<uint32_t> readInputData32()
{
	std::ifstream inputFile("input.txt", std::ios::binary);

	if (!inputFile.is_open())
	{
		std::cout << "Unable to open input.txt" << std::endl;
		throw std::runtime_error("unable to open input.txt");
	}

	inputFile.seekg(0, std::ios::end);
	std::streamsize size = inputFile.tellg();

	std::cout << std::format("input.txt size: {}\n", size);

	inputFile.seekg(0, std::ios::beg);

	// Use a std::string buffer so iterators work natively with utf8cpp
	std::string buffer(size, '\0');
	if (!inputFile.read(buffer.data(), size))
	{
		throw std::runtime_error("Failed to read file input.txt");
	}

	std::vector<uint32_t> code_points;
	try
	{
		// Safely decode raw UTF-8 bytes into 32-bit Unicode code points
		utf8::utf8to32(buffer.begin(), buffer.end(), std::back_inserter(code_points));
	}
	catch (const std::exception& e)
	{
		std::cout << std::format("UTF-8 decoding error: {}\n", e.what());
		throw;
	}

	std::cout << std::format("Successfully decoded {} code points.\n", code_points.size());
	return code_points;
}

