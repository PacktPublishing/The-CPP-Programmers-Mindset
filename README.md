# Computational-Thinking-with-CPP

## Description

C++23 projects

## Table of contents

- [Build applications on Ubuntu platform](#ubuntu-platform)
    - [Ubuntu platform native build](#ubuntu-platform-native-build)
- [Build applications on Mac platform](#mac-platform)
    - [Mac platform native build](#mac-platform-native-build)
- [Build applications on Windows 10/11 platform](#windows-platform)
    - [Windows 10/11 platform Ubuntu on WSL2 build](#windows-platform-wsl2-build)
- [Recommended Cross-Platform IDE for C++23](#recommended-cross-platform-ide)
    - [CLion: A Cross-Platform IDE for C and C++ by JetBrains](#clion-cross-platform-ide)
    - [Visual Studio Code: A Cross-Platform Code Editor by Microsoft](#vscode-cross-platform-ide)

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

## <a name="windows-platform">Build applications on Windows 10/11 platform</a>

There are two options to build applications on Windows 10/11 platform: Ubuntu on
WSL2 build and build in a Docker environment.

### <a name="windows-platform-wsl2-build">Windows 10/11 platform Ubuntu on WSL2 build</a>

* [Install Windows Terminal](https://learn.microsoft.com/en-us/windows/terminal/install "Windows Terminal")

* [Install Ubuntu on WSL2](https://learn.microsoft.com/en-us/windows/wsl/install "Ubuntu on WSL2")

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

## <a name="recommended-cross-platform-ide">Recommended Cross-Platform IDE for C++23</a>

### <a name="clion-cross-platform-ide">CLion: A Cross-Platform IDE for C and C++ by JetBrains</a>

* [CLion](https://www.jetbrains.com/clion/)

### <a name="vscode-cross-platform-ide">Visual Studio Code: A Cross-Platform Code Editor by Microsoft</a>

* [Visual Studio Code](https://code.visualstudio.com/)
