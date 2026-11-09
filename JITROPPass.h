#pragma once

#include "llvm/IR/PassManager.h"
#include "llvm/Passes/PassPlugin.h"
#include "llvm/Support/CommandLine.h"

struct JITROPPass : llvm::PassInfoMixin<JITROPPass>
{
    llvm::PreservedAnalyses run(llvm::Module& m, llvm::ModuleAnalysisManager&);
};

bool RunJITROP(llvm::Module& m);

extern llvm::cl::opt<bool> JITROPEnable;