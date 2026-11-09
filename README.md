# JIT Ropfuscator
Built for LLVM 19.1 - Windows x64 <br> <br>

Same concept as [ROPFuscator](https://github.com/IntelSDM/ROPFuscatorLLVM), without the known gadgets system. We JIT the gadgets at runtime instead. Making it quite a bit less secure. <br>
The intention is for where you can't easily deploy per Windows build applications. Such as in a preloader. Still prevents faults pointing to your code, still CET compliant. <br>
Handles Read/Write/Calls.<br>
<br>
Places a gadget scanner, Memory Mapper and then creates the stubs in RWX allocated memory. <br>

<img width="894" height="710" alt="image" src="https://github.com/user-attachments/assets/b9d019cd-4530-425e-b7ef-14747c3a888f" />

## Annotations
```c
__attribute__((annotate("jitrop")))
``` 
* Activates ROP gadget usage where it can be used.

## How To Build
* Download [LLVM 19.1](https://github.com/llvm/llvm-project/releases/tag/llvmorg-19.1.0)
* Download [VCPKG](https://github.com/microsoft/vcpkg)
* VCPKG install asmjit
* Open build.bat in a text editor.
* Find the LLVM 19.1 references in the command line, Change them to your file path.
* Find the VCPKG references in the command line, Change them to your file path.
* Run the build script
* Build your application with the ``` -load-pass-plugin ``` Param to the FunctionPass.dll
