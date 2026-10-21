
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${TORCH_CXX_FLAGS}")

set (MY_SOURCES
	"${CMAKE_SOURCE_DIR}/Tokenizer/source/main.cpp"
)

set(MY_HEADERS
	"${CMAKE_SOURCE_DIR}/include/tokenizer.h"
	"${CMAKE_SOURCE_DIR}/include/utils.h"
)

source_group("Header Files" FILES ${MY_HEADERS})

include(FetchContent)
FetchContent_Declare(
  utf8cpp
  GIT_REPOSITORY https://github.com/nemtrif/utfcpp
  GIT_TAG        v4.0.5
)
FetchContent_MakeAvailable(utf8cpp)

# Explicitly create the target if utf8cpp's internal CMake doesn't expose it
if(NOT TARGET utf8cpp::utf8cpp)
  add_library(utf8cpp::utf8cpp INTERFACE IMPORTED)
  set_target_properties(utf8cpp::utf8cpp PROPERTIES
    INTERFACE_INCLUDE_DIRECTORIES "${utf8cpp_SOURCE_DIR}/source"
  )
endif()

add_executable(Tokenizer 
	${MY_SOURCES} 
)

target_sources(Tokenizer PRIVATE 
	${MY_SOURCES} 
    ${MY_HEADERS}
)

target_include_directories(Tokenizer PRIVATE 
	${CMAKE_SOURCE_DIR}/include
)

target_link_libraries(Tokenizer PRIVATE utf8cpp::utf8cpp)

target_compile_features(Tokenizer PRIVATE cxx_std_20)

if(WIN32)
	target_compile_features(Tokenizer PRIVATE cxx_std_20)
	target_compile_options(Tokenizer PRIVATE "/utf-8")
	add_custom_command(TARGET Tokenizer POST_BUILD

		COMMAND ${CMAKE_COMMAND} -E copy_if_different "${CMAKE_SOURCE_DIR}/TransformerC/source/input.txt" "${CMAKE_BINARY_DIR}/x64/$<CONFIG>"
	)    
	set_target_properties(Tokenizer PROPERTIES
    		RUNTIME_OUTPUT_DIRECTORY "${CMAKE_SOURCE_DIR}/build/x64/$<CONFIG>"
    		#RUNTIME_OUTPUT_DIRECTORY_DEBUG "${CMAKE_SOURCE_DIR}/build/bin"
    		#RUNTIME_OUTPUT_DIRECTORY_RELEASE "${CMAKE_SOURCE_DIR}/build/bin"
		CXX_STANDARD 20
    		CXX_STANDARD_REQUIRED ON
    	CXX_EXTENSIONS OFF
)

elseif(LINUX)
	set(CMAKE_CXX_STANDARD 20)
	set(CMAKE_CXX_STANDARD_REQUIRED ON)
	set(CMAKE_CXX_EXTENSIONS OFF)


	set_target_properties(Tokenizer PROPERTIES
	    RUNTIME_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}/bin"
	    RUNTIME_OUTPUT_DIRECTORY_DEBUG "${CMAKE_BINARY_DIR}/bin"
	    RUNTIME_OUTPUT_DIRECTORY_RELEASE "${CMAKE_BINARY_DIR}/bin"
	)
endif()





