
#include <iostream>
#include <string>

#ifdef _WIN32
#include <windows.h>
#endif

#include <utf8.h>

int main()
{
#ifdef _WIN32
    SetConsoleOutputCP(CP_UTF8);
#endif

    std::string text = "Привет, мир! ✓ ★";

    size_t count = utf8::distance(text.begin(), text.end());

    std::cout << text << "\n";
    std::cout << "Character Count: " << count << "\n";

    return 0;
}