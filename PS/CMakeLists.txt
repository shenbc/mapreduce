cmake_minimum_required(VERSION 3.10)
project(PS)

set(CMAKE_CXX_STANDARD 11)

add_executable(PS server/main.cpp common/NGAPacket.h server/main.h)
