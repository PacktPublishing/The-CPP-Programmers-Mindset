# Computational-Thinking-with-CPP

## Description

C++23 projects

### The recommended build platform for C++23 code is the Ubuntu 24.04 (native build or build in a Docker environment).

## Table of contents

- [Build applications on Ubuntu platform](#ubuntu-platform)
    - [Ubuntu platform native build](#ubuntu-platform-native-build)
    - [Ubuntu platform build in a Docker environment](#ubuntu-platform-docker-environment)
- [Build applications on Mac platform](#mac-platform)
    - [Mac platform native build](#mac-platform-native-build)
    - [Mac platform build in a Docker environment](#mac-platform-docker-environment)
- [Build applications on Windows 10/11 platform](#windows-platform)
    - [Windows 10/11 platform Ubuntu on WSL2 build](#windows-platform-wsl2-build)
    - [Windows 10/11 platform build in a Docker environment](#windows-platform-docker-environment)
- [Remove All Docker Images](#remove-all-docker-images)
- [Recommended C++ Style Guides](#recommended-c++-style-guides)
- [Recommended C++ code formatter](#recommended-c++-code-formatter)
- [Recommended Cross-Platform IDE for C++23](#recommended-cross-platform-ide)
    - [CLion: A Cross-Platform IDE for C and C++ by JetBrains](#clion-cross-platform-ide)
    - [Visual Studio Code: A Cross-Platform Code Editor by Microsoft](#vscode-cross-platform-ide)
- [GoogleTest: Google's C++ test framework](#google-c++-test-framework)

## <a name="ubuntu-platform">Build applications on Ubuntu platform</a>

There are two options to build applications on the Ubuntu platform: native build
and build in a Docker environment.

### <a name="ubuntu-platform-native-build">Ubuntu platform native build</a>

The recommended Ubuntu version is the Ubuntu 24.04.

* Install Git and clone the GitHub repository

```bash
sudo apt update
sudo apt upgrade
sudo apt autoremove
sudo apt install git
git clone https://github.com/PacktPublishing/Computational-Thinking-with-CPP.git
```

* Install the project dependencies

```bash
cd tools
sudo ./prerequisites-ubuntu.sh
cd ..
```

* Build the applications in folder with the `CMakeLists.txt` file

```bash
cmake -B ./build
cmake --build ./build
```

* Run the application

```bash
cd build
./application_name
```

### <a name="ubuntu-platform-docker-environment">Ubuntu platform build in a Docker environment</a>

* Install Git and clone the GitHub repository

```bash
sudo apt update
sudo apt upgrade
sudo apt autoremove
sudo apt install git
git clone https://github.com/PacktPublishing/Computational-Thinking-with-CPP.git
```

* [Install Docker Engine on Ubuntu](https://docs.docker.com/engine/install/ubuntu/)

* Build `Computational-Thinking-with-CPP` Docker image

```bash
cd tools
./build-docker-image.sh
cd ..
```

* Run `Computational-Thinking-with-CPP` Docker container

```bash
./run-docker-image.sh
```

* Build the applications in folder with the `CMakeLists.txt` file

```bash
cmake -B ./build
cmake --build ./build
```

* Run the application

```bash
./application_name
```

* Exit `Computational-Thinking-with-CPP` Docker container

```bash
exit
```

## <a name="mac-platform">Build applications on Mac platform</a>

There are two options to build applications on Mac platform: native build and
build in a Docker environment.

### <a name="mac-platform-native-build">Mac platform native build</a>

* Install Xcode Command Line Tools on a Mac

```bash
xcode-select --install
```

* [Install Homebrew on Mac](https://brew.sh/)

* Update and Check the Homebrew repository

```bash
brew update
brew upgrade
brew autoremove
brew cleanup
brew doctor
```

* Install Git and clone the GitHub repository

```bash
brew install git
git clone https://github.com/PacktPublishing/Computational-Thinking-with-CPP.git
```

* Install the project dependencies

```bash
cd tools
./prerequisites-mac.sh
cd ..
```

* Build the applications in folder with the `CMakeLists.txt` file

```bash
cmake -B ./build
cmake --build ./build
```

* Run the application

```bash
cd build
./application_name
```

### <a name="mac-platform-docker-environment">Mac platform build in a Docker environment</a>

* [Install Homebrew on Mac](https://brew.sh/)

* Update and Check the Homebrew repository

```bash
brew update
brew upgrade
brew autoremove
brew cleanup
brew doctor
```

* Install Git and clone the GitHub repository

```bash
brew install git
git clone https://github.com/PacktPublishing/Computational-Thinking-with-CPP.git
```

* [Install Docker Desktop on Mac](https://docs.docker.com/desktop/setup/install/mac-install/)

* Run Docker Desktop

* Build `Computational-Thinking-with-CPP` Docker image

```bash
cd tools
./build-docker-image.sh
cd ..
```

* Run `Computational-Thinking-with-CPP` Docker container

```bash
./run-docker-image.sh
```

* Build the applications in folder with the `CMakeLists.txt` file

```bash
cmake -B ./build
cmake --build ./build
```

* Run the application

```bash
./application_name
```

* Exit `Computational-Thinking-with-CPP` Docker container

```bash
exit
```

## <a name="windows-platform">Build applications on Windows 10/11 platform</a>

There are two options to build applications on Windows 10/11 platform: Ubuntu on
WSL2 build and build in a Docker environment.

### <a name="windows-platform-wsl2-build">Windows 10/11 platform Ubuntu on WSL2 build</a>

* [Install Windows Terminal](https://learn.microsoft.com/en-us/windows/terminal/install)

* [Install Ubuntu on WSL2](https://learn.microsoft.com/en-us/windows/wsl/install)

* Run Ubuntu on WSL2

* Install Git and clone the GitHub repository

```bash
sudo apt update
sudo apt upgrade
sudo apt autoremove
sudo apt install git
git clone https://github.com/PacktPublishing/Computational-Thinking-with-CPP.git
```

* Install the project dependencies

```bash
cd tools
sudo ./prerequisites-ubuntu.sh
cd ..
```

* Build the applications in folder with the `CMakeLists.txt` file

```bash
cmake -B ./build
cmake --build ./build
```

* Run the application

```bash
cd build
./application_name
```

### <a name="windows-platform-docker-environment">Windows 10/11 platform build in a Docker environment</a>

* [Install Windows Terminal](https://learn.microsoft.com/en-us/windows/terminal/install)

* [Install Git for Windows](https://git-scm.com/downloads/win)

* Clone the GitHub repository

```bash
git clone https://github.com/PacktPublishing/Computational-Thinking-with-CPP.git
```

* [Install Docker Desktop on Windows](https://docs.docker.com/desktop/install/windows-install/)

* Run Docker Desktop

* Build `Computational-Thinking-with-CPP` Docker image

```bash
cd tools
docker build -t Computational-Thinking-with-CPP .
cd ..
```

* Run `Computational-Thinking-with-CPP` Docker container

```bash
docker run --rm -it --volume "$(pwd):/root" Computational-Thinking-with-CPP:latest
```

* Build the applications in folder with the `CMakeLists.txt` file

```bash
cmake -B ./build
cmake --build ./build
```

* Run the application

```bash
cd build
./application_name
```

* Exit `Computational-Thinking-with-CPP` Docker container

```bash
exit
```

## <a name="remove-all-docker-images">Remove All Docker Images</a>

Remove any stopped containers and all unused images

```bash
cd tools
./remove-docker-images.sh
cd ..
```

## <a name="recommended-c++-style-guides">Recommended C++ Style Guides</a>

1. [Google C++ Style Guide](https://google.github.io/styleguide/cppguide.html)
2. [C++ Core Guidelines](https://isocpp.github.io/CppCoreGuidelines/CppCoreGuidelines)

`cpplint` - static code checker for C++. `cpplint` is a command-line tool to
check C/C++ files for style issues following Google's C++ style guide.

`clang-tidy` is a clang-based C++ “linter” tool. Its purpose is to provide an
extensible framework for diagnosing and fixing typical programming errors, like
style violations, interface misuse, or bugs that can be deduced via static
analysis.

## <a name="recommended-c++-code-formatter">Recommended C++ code formatter</a>

`clang-format` is a widely-used C++ code formatter. It provides an option to
define code style options in YAML-formatted files — named `.clang-format` or
`_clang-format` — and these files often become a part of your project where you
keep all code style rules. It is highly recommended to format your changed C++
code before opening pull requests, which will save you and the reviewers' time.

## <a name="recommended-cross-platform-ide">Recommended Cross-Platform IDE for C++23</a>

### <a name="clion-cross-platform-ide">CLion: A Cross-Platform IDE for C and C++ by JetBrains</a>

* [CLion](https://www.jetbrains.com/clion/)
* [CLion Docker toolchain](https://www.jetbrains.com/help/clion/clion-toolchains-in-docker.html)
* [CLion ClangFormat code formatter](https://www.jetbrains.com/help/clion/clangformat-as-alternative-formatter.html)

### <a name="vscode-cross-platform-ide">Visual Studio Code: A Cross-Platform Code Editor by Microsoft</a>

* [Visual Studio Code](https://code.visualstudio.com/)
* [VSCode Developing inside a Container](https://code.visualstudio.com/docs/devcontainers/containers)
* [VSCode ClangFormat code formatter](https://code.visualstudio.com/docs/cpp/cpp-ide)

## <a name="google-c++-test-framework">GoogleTest: Google's C++ test framework</a>

* [GoogleTest](https://github.com/google/googletest)
* [GoogleTest User’s Guide](https://google.github.io/googletest/)
* [GTest Reference](docs/GTestReference.png)
  ![GTest Reference](docs/GTestReference.png)
