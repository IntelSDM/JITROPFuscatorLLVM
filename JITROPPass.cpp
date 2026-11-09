#include "JITROPPass.h"

#include "llvm/ADT/DenseSet.h"
#include "llvm/ADT/StringRef.h"
#include "llvm/IR/BasicBlock.h"
#include "llvm/IR/Constants.h"
#include "llvm/IR/DataLayout.h"
#include "llvm/IR/DerivedTypes.h"
#include "llvm/IR/Function.h"
#include "llvm/IR/GlobalVariable.h"
#include "llvm/IR/IRBuilder.h"
#include "llvm/IR/Instructions.h"
#include "llvm/IR/IntrinsicInst.h"
#include "llvm/IR/Module.h"
#include "llvm/Support/CommandLine.h"
#include "llvm/Transforms/Utils/ModuleUtils.h"

#include <algorithm>
#include <cstdint>
#include <string>
#include <unordered_map>
#include <vector>

using namespace llvm;

llvm::cl::opt<bool> JITROPEnable("obfjitrop", llvm::cl::init(true), llvm::cl::desc("MasterToggle"));

static constexpr unsigned CacheSlots = 4096;
static constexpr unsigned GadgetBufSize = 16;
static constexpr unsigned InitCtorPriority = 101;

enum RegByteField
{
    FldMovREX = 0,
    FldMovOp = 1,
    FldStREX = 2,
    FldStMR0 = 3,
    FldStMR8 = 4,
    FldJmpB0 = 5,
    FldJmpB1 = 6,
    FldJmpB2 = 7,
};

enum CallShape
{
    ShapeBasic,
    ShapeShadow,
    ShapeCopy,
};

struct ChainSelectors
{
    Value* C;
    Value* T;
    Value* B;
    Value* Sub;
    Value* Save;
    Value* Order;
    Value* SaveReg;
    Value* Xfer;
};

struct CopyBytes
{
    Value* LoadREX;
    Value* LoadModRM;
    Value* StoreREX;
    Value* StoreModRM;
};

struct CallSite
{
    CallInst* Call;
    uint64_t Sid;
};

struct OpSite
{
    enum Kind { KLoad, KStore, KLea } Kind;
    Instruction* Inst;
    Use* LeaUse;
    GlobalVariable* Target;
    unsigned SizeBytes;
    Type* ValTy;
    uint64_t Sid;
};

struct RuntimeRefs
{
    StructType* CacheEntryTy;
    ArrayType* CacheArrTy;
    GlobalVariable* GvBareRet;
    GlobalVariable* GvTramp40;
    GlobalVariable* GvTramp56;
    GlobalVariable* GvTramp72;
    GlobalVariable* GvTramp88;
    GlobalVariable* GvCacheBasic;
    GlobalVariable* GvCacheShadow;
    GlobalVariable* GvCacheCopy;
    GlobalVariable* GvCacheLoad;
    GlobalVariable* GvCacheStore;
    GlobalVariable* GvCacheLea;
    GlobalVariable* GvNtdllName;
    GlobalVariable* GvRegBytes;
    GlobalVariable* GvSaveBytes;
    GlobalVariable* GvCopyBytes;
    GlobalVariable* GvCalleeTable;
    GlobalVariable* GvAddrTable;
    GlobalVariable* GvSiteCache;
    Function* FnInitTables;
    FunctionCallee FnVAlloc;
    FunctionCallee FnGMH;
    Function* FnFindBareRet;
    Function* FnFindTramp40;
    Function* FnFindTramp56;
    Function* FnFindTramp72;
    Function* FnFindTramp88;
    Function* FnBuildChainBasic;
    Function* FnBuildChainShadow;
    Function* FnBuildChainCopy;
    Function* FnGetChainBasic;
    Function* FnGetChainShadow;
    Function* FnGetChainCopy;
    Function* FnBuildLoadGadget;
    Function* FnBuildStoreGadget;
    Function* FnBuildLeaGadget;
    Function* FnGetLoadGadget;
    Function* FnGetStoreGadget;
    Function* FnGetLeaGadget;
    uint64_t K1;
    uint64_t K2;
    std::unordered_map<Function*, unsigned> CalleeIndex;
    std::unordered_map<GlobalVariable*, unsigned> AddrIndex;
    DenseSet<GlobalVariable*> JitropGvs;
    DenseSet<Function*> JitropFns;
    DenseSet<Function*> NoJitropFns;
};

static uint64_t Djb2Hash(const char* s)
{
    uint64_t h = 5381;
    while (*s)
    {
        h = ((h << 5) + h) + (uint64_t)(uint8_t)(*s);
        ++s;
    }
    return h;
}

static void CollectAnnotations(Module& m, RuntimeRefs& r)
{
    GlobalVariable* ga = m.getGlobalVariable("llvm.global.annotations");
    // GA may be a declaration only forward reference with no initializer
    if (!ga || !ga->hasInitializer()) return;
    auto* ca = dyn_cast<ConstantArray>(ga->getInitializer());
    if (!ca) return;
    for (unsigned i = 0, e = ca->getNumOperands(); i < e; ++i)
    {
        auto* cs = dyn_cast<ConstantStruct>(ca->getOperand(i));
        if (!cs || cs->getNumOperands() < 2) continue;
        auto* tgtc = cs->getOperand(0);
        while (auto* ce = dyn_cast<ConstantExpr>(tgtc))
        {
            if (ce->getOpcode() == Instruction::BitCast ||
                ce->getOpcode() == Instruction::AddrSpaceCast)
                tgtc = ce->getOperand(0);
            else break;
        }
        auto* strgv = dyn_cast<GlobalVariable>(cs->getOperand(1)->stripPointerCasts());
        if (!strgv || !strgv->hasInitializer()) continue;
        auto* cds = dyn_cast<ConstantDataArray>(strgv->getInitializer());
        if (!cds || !cds->isCString()) continue;
        StringRef s = cds->getAsCString();
        if (s == "jitrop")
        {
            if (auto* gv = dyn_cast<GlobalVariable>(tgtc)) r.JitropGvs.insert(gv);
            else if (auto* fn = dyn_cast<Function>(tgtc)) r.JitropFns.insert(fn);
        }
        else if (s == "nojitrop")
        {
            if (auto* fn = dyn_cast<Function>(tgtc)) r.NoJitropFns.insert(fn);
        }
    }
}

static bool IsOurRuntime(StringRef n)
{
    return n.starts_with("JITROP");
}

static bool FunctionIsSkippable(Function& f, const RuntimeRefs& r)
{
    if (f.isDeclaration()) return true;
    if (IsOurRuntime(f.getName())) return true;
    if (f.getName().starts_with("__cond_stomp_")) return true;
    if (r.NoJitropFns.count(&f)) return true;
    return false;
}

static uint64_t ComputeSiteId(Function& f, unsigned idx, char tag)
{
    std::string s = "JITROP:";
    s += tag;
    s += ':';
    s += f.getName().str();
    s += ':';
    s += std::to_string(idx);
    uint64_t h = Djb2Hash(s.c_str());
    return h == 0 ? 1 : h;
}

static std::pair<uint64_t, uint64_t> ComputeKeys(Module& m)
{
    std::string base = m.getName().str();
    uint64_t k1 = Djb2Hash(("JITROPK1:" + base).c_str());
    uint64_t k2 = Djb2Hash(("JITROPK2:" + base).c_str());
    if (k1 == 0) k1 = 0xDEADBEEFCAFEBABEULL;
    if (k2 == 0) k2 = 0x1337F00DFACEC0DEULL;
    return {k1, k2};
}

static uint64_t SafeEncSid(uint64_t pt, uint64_t k1)
{
    // bug: enc == 0 collides with the cache empty marker
    uint64_t enc = pt ^ k1;
    return enc == 0 ? pt : enc;
}

static bool CallSiteEligible(CallInst* ci, const RuntimeRefs& r)
{
    if (ci->isMustTailCall()) return false;
    if (ci->isInlineAsm()) return false;
    if (ci->hasOperandBundles()) return false;
    Function* callee = ci->getCalledFunction();
    if (!callee) return false;
    if (callee->isIntrinsic()) return false;
    if (callee->isVarArg()) return false;
    StringRef n = callee->getName();
    if (IsOurRuntime(n)) return false;
    if (n.starts_with("__cond_stomp_")) return false;
    if (n == "GetModuleHandleA") return false;
    if (n == "VirtualAlloc") return false;
    if (r.NoJitropFns.count(callee)) return false;
    return true;
}

static bool IsSupportedScalarType(Type* t, const DataLayout&, unsigned& sz)
{
    if (t->isIntegerTy())
    {
        unsigned bits = t->getIntegerBitWidth();
        if (bits == 8 || bits == 16 || bits == 32 || bits == 64)
        {
            sz = bits / 8;
            return true;
        }
        return false;
    }
    if (t->isFloatTy()) { sz = 4; return true; }
    if (t->isDoubleTy()) { sz = 8; return true; }
    if (t->isPointerTy()) { sz = 8; return true; }
    return false;
}

static void WriteConst(IRBuilder<>& b, Value* buf, Value* off, uint8_t v)
{
    Type* i8 = Type::getInt8Ty(b.getContext());
    Value* p = b.CreateGEP(i8, buf, off);
    b.CreateAlignedStore(ConstantInt::get(i8, v), p, Align(1));
}

static void WriteVar(IRBuilder<>& b, Value* buf, Value* off, Value* v)
{
    Type* i8 = Type::getInt8Ty(b.getContext());
    Value* p = b.CreateGEP(i8, buf, off);
    b.CreateAlignedStore(v, p, Align(1));
}

static void Write64(IRBuilder<>& b, Value* buf, Value* off, Value* v)
{
    Type* i8 = Type::getInt8Ty(b.getContext());
    Value* p = b.CreateGEP(i8, buf, off);
    b.CreateAlignedStore(v, p, Align(1));
}

static Value* AddOff(IRBuilder<>& b, Value* base, uint64_t k)
{
    return b.CreateAdd(base, ConstantInt::get(Type::getInt64Ty(b.getContext()), k));
}

static Value* RegByte(IRBuilder<>& b, GlobalVariable* tbl, Value* sel, unsigned fld)
{
    Type* i8 = Type::getInt8Ty(b.getContext());
    Type* i64 = Type::getInt64Ty(b.getContext());
    Value* off = b.CreateAdd(b.CreateMul(sel, ConstantInt::get(i64, 8)),
                              ConstantInt::get(i64, fld));
    Value* p = b.CreateGEP(i8, tbl, off);
    return b.CreateAlignedLoad(i8, p, Align(1));
}

static ChainSelectors ComputeChainSelectors(IRBuilder<>& b, Value* sid, Type* i64)
{
    Value* three = ConstantInt::get(i64, 3);
    Value* nine = ConstantInt::get(i64, 9);
    Value* twentyseven = ConstantInt::get(i64, 27);
    Value* two = ConstantInt::get(i64, 2);
    Value* four = ConstantInt::get(i64, 4);

    ChainSelectors s{};
    s.C = b.CreateURem(sid, three, "SelC");
    s.T = b.CreateURem(b.CreateUDiv(sid, three), three, "SelT");
    s.B = b.CreateURem(b.CreateUDiv(sid, nine), three, "SelB");

    Value* pool = b.CreateUDiv(sid, twentyseven);
    s.Sub = b.CreateURem(pool, four, "SelSub");
    Value* p2 = b.CreateUDiv(pool, four);
    s.Save = b.CreateURem(p2, two, "SelSave");
    Value* p3 = b.CreateUDiv(p2, two);
    s.Order = b.CreateURem(p3, two, "SelOrder");
    Value* p4 = b.CreateUDiv(p3, two);
    s.SaveReg = b.CreateURem(p4, two, "SelSaveReg");
    Value* p5 = b.CreateUDiv(p4, two);
    s.Xfer = b.CreateURem(p5, two, "SelXfer");
    return s;
}

static Value* EmitSubRSP(IRBuilder<>& b, Module& m, Value* buf, Value* startoff, Value* selsub, Value* nval)
{
    LLVMContext& c = m.getContext();
    Type* i8 = Type::getInt8Ty(c);
    Type* i64 = Type::getInt64Ty(c);
    Value* nu8 = b.CreateTrunc(nval, i8);
    Value* negn8 = b.CreateTrunc(b.CreateSub(ConstantInt::get(i64, 0), nval), i8);

    Function* f = b.GetInsertBlock()->getParent();
    BasicBlock* a0 = BasicBlock::Create(c, "SubA0", f);
    BasicBlock* a1 = BasicBlock::Create(c, "SubA1", f);
    BasicBlock* a2 = BasicBlock::Create(c, "SubA2", f);
    BasicBlock* a3 = BasicBlock::Create(c, "SubA3", f);
    BasicBlock* done = BasicBlock::Create(c, "SubDone", f);

    SwitchInst* sw = b.CreateSwitch(selsub, a0, 3);
    sw->addCase(ConstantInt::get(cast<IntegerType>(selsub->getType()), 1), a1);
    sw->addCase(ConstantInt::get(cast<IntegerType>(selsub->getType()), 2), a2);
    sw->addCase(ConstantInt::get(cast<IntegerType>(selsub->getType()), 3), a3);

    b.SetInsertPoint(a0);
    // sub rsp, N  (imm8)   48 83 EC N
    WriteConst(b, buf, startoff, 0x48);                 // REX.W
    WriteConst(b, buf, AddOff(b, startoff, 1), 0x83);   // SUB r/m64, imm8 opcode
    WriteConst(b, buf, AddOff(b, startoff, 2), 0xEC);   // ModRM /5 rsp
    WriteVar(b, buf, AddOff(b, startoff, 3), nu8);      // imm8 = N
    b.CreateBr(done);

    b.SetInsertPoint(a1);
    // add rsp, -N  (imm8)  48 83 C4 -N
    WriteConst(b, buf, startoff, 0x48);                 // REX.W
    WriteConst(b, buf, AddOff(b, startoff, 1), 0x83);   // ADD r/m64, imm8 opcode
    WriteConst(b, buf, AddOff(b, startoff, 2), 0xC4);   // ModRM /0 rsp
    WriteVar(b, buf, AddOff(b, startoff, 3), negn8);    // imm8 = -N
    b.CreateBr(done);

    b.SetInsertPoint(a2);
    // lea rsp, [rsp - N]   48 8D 64 24 -N
    WriteConst(b, buf, startoff, 0x48);                 // REX.W
    WriteConst(b, buf, AddOff(b, startoff, 1), 0x8D);   // LEA opcode
    WriteConst(b, buf, AddOff(b, startoff, 2), 0x64);   // ModRM rsp [SIB+disp8]
    WriteConst(b, buf, AddOff(b, startoff, 3), 0x24);   // SIB (base=rsp)
    WriteVar(b, buf, AddOff(b, startoff, 4), negn8);    // disp8 = -N
    b.CreateBr(done);

    b.SetInsertPoint(a3);
    // sub rsp, N  (imm32)  48 81 EC N 00 00 00
    WriteConst(b, buf, startoff, 0x48);                 // REX.W
    WriteConst(b, buf, AddOff(b, startoff, 1), 0x81);   // SUB r/m64, imm32 opcode
    WriteConst(b, buf, AddOff(b, startoff, 2), 0xEC);   // ModRM /5 rsp
    WriteVar(b, buf, AddOff(b, startoff, 3), nu8);      // imm32 low byte  = N
    WriteConst(b, buf, AddOff(b, startoff, 4), 0x00);   // imm32 byte 1
    WriteConst(b, buf, AddOff(b, startoff, 5), 0x00);   // imm32 byte 2
    WriteConst(b, buf, AddOff(b, startoff, 6), 0x00);   // imm32 byte 3
    b.CreateBr(done);

    b.SetInsertPoint(done);
    PHINode* sz = b.CreatePHI(i64, 4, "SubSize");
    sz->addIncoming(ConstantInt::get(i64, 4), a0);
    sz->addIncoming(ConstantInt::get(i64, 4), a1);
    sz->addIncoming(ConstantInt::get(i64, 5), a2);
    sz->addIncoming(ConstantInt::get(i64, 7), a3);
    return b.CreateAdd(startoff, sz);
}

static Value* EmitSaveMov(IRBuilder<>& b, GlobalVariable* savetbl, Value* buf, Value* startoff, Value* selsave, Value* selsavereg)
{
    Type* i8 = Type::getInt8Ty(b.getContext());
    Type* i64 = Type::getInt64Ty(b.getContext());

    Value* row = b.CreateAdd(b.CreateMul(selsave, ConstantInt::get(i64, 2)), selsavereg);
    Value* rowoff = b.CreateMul(row, ConstantInt::get(i64, 3));

    Value* p0 = b.CreateGEP(i8, savetbl, rowoff);
    Value* p1 = b.CreateGEP(i8, savetbl, b.CreateAdd(rowoff, ConstantInt::get(i64, 1)));
    Value* p2 = b.CreateGEP(i8, savetbl, b.CreateAdd(rowoff, ConstantInt::get(i64, 2)));
    Value* b0 = b.CreateAlignedLoad(i8, p0, Align(1));
    Value* b1 = b.CreateAlignedLoad(i8, p1, Align(1));
    Value* b2 = b.CreateAlignedLoad(i8, p2, Align(1));

    WriteVar(b, buf, startoff, b0);
    WriteVar(b, buf, AddOff(b, startoff, 1), b1);
    WriteVar(b, buf, AddOff(b, startoff, 2), b2);
    return AddOff(b, startoff, 3);
}

static CopyBytes ReadCopyBytes(IRBuilder<>& b, GlobalVariable* copytbl, Value* selsavereg, Value* selxfer)
{
    Type* i8 = Type::getInt8Ty(b.getContext());
    Type* i64 = Type::getInt64Ty(b.getContext());

    Value* row = b.CreateAdd(b.CreateMul(selsavereg, ConstantInt::get(i64, 2)), selxfer);
    Value* rowoff = b.CreateMul(row, ConstantInt::get(i64, 4));

    auto ld = [&](uint64_t k)
    {
        Value* p = b.CreateGEP(i8, copytbl, b.CreateAdd(rowoff, ConstantInt::get(i64, k)));
        return b.CreateAlignedLoad(i8, p, Align(1));
    };
    CopyBytes cb;
    cb.LoadREX = ld(0);
    cb.LoadModRM = ld(1);
    cb.StoreREX = ld(2);
    cb.StoreModRM = ld(3);
    return cb;
}

static Value* EmitLoadStoreSlot(IRBuilder<>& b, GlobalVariable* tbl, Value* buf, Value* startoff, Value* immval, Value* sel, bool disp8)
{
    // mov <reg>, imm64   REX(48/49) + opcode(B8+r)  then 8 imm bytes
    WriteVar(b, buf, startoff, RegByte(b, tbl, sel, FldMovREX));
    WriteVar(b, buf, AddOff(b, startoff, 1), RegByte(b, tbl, sel, FldMovOp));
    Write64(b, buf, AddOff(b, startoff, 2), immval);
    // mov [rsp(+8)], <reg>   REX + 89 + ModRM + SIB (0x24) [+ disp8]
    WriteVar(b, buf, AddOff(b, startoff, 10), RegByte(b, tbl, sel, FldStREX));
    WriteConst(b, buf, AddOff(b, startoff, 11), 0x89);   // MOV r/m64, r64 opcode
    if (!disp8)
    {
        WriteVar(b, buf, AddOff(b, startoff, 12), RegByte(b, tbl, sel, FldStMR0));
        WriteConst(b, buf, AddOff(b, startoff, 13), 0x24);  // SIB base=rsp
        return AddOff(b, startoff, 14);
    }
    WriteVar(b, buf, AddOff(b, startoff, 12), RegByte(b, tbl, sel, FldStMR8));
    WriteConst(b, buf, AddOff(b, startoff, 13), 0x24);      // SIB base=rsp
    WriteConst(b, buf, AddOff(b, startoff, 14), 0x08);      // disp8 = +8
    return AddOff(b, startoff, 15);
}

static Value* EmitLoadJmpSlot(IRBuilder<>& b, GlobalVariable* tbl, Value* buf, Value* startoff, Value* immval, Value* sel)
{
    // mov <reg>, imm64   REX + B8+r + 8 imm bytes
    WriteVar(b, buf, startoff, RegByte(b, tbl, sel, FldMovREX));
    WriteVar(b, buf, AddOff(b, startoff, 1), RegByte(b, tbl, sel, FldMovOp));
    Write64(b, buf, AddOff(b, startoff, 2), immval);
    // jmp <reg>   rax: FF E0 00 pad | r10/r11: 41 FF E2/E3  (uniform 3 bytes)
    WriteVar(b, buf, AddOff(b, startoff, 10), RegByte(b, tbl, sel, FldJmpB0));
    WriteVar(b, buf, AddOff(b, startoff, 11), RegByte(b, tbl, sel, FldJmpB1));
    WriteVar(b, buf, AddOff(b, startoff, 12), RegByte(b, tbl, sel, FldJmpB2));
    return AddOff(b, startoff, 13);
}

static Value* EmitTailPair(IRBuilder<>& b, GlobalVariable* tbl, Value* buf, Value* startoff, Value* callee, Value* selc, Value* tramp, Value* selt, Value* selorder)
{
    LLVMContext& c = b.getContext();
    Type* i64 = Type::getInt64Ty(c);

    Function* f = b.GetInsertBlock()->getParent();
    BasicBlock* c0 = BasicBlock::Create(c, "OrdC0", f);
    BasicBlock* c1 = BasicBlock::Create(c, "OrdC1", f);
    BasicBlock* done = BasicBlock::Create(c, "OrdDone", f);
    b.CreateCondBr(b.CreateICmpEQ(selorder, ConstantInt::get(i64, 0)), c0, c1);

    b.SetInsertPoint(c0);
    Value* mid0 = EmitLoadStoreSlot(b, tbl, buf, startoff, callee, selc, false);
    (void)EmitLoadStoreSlot(b, tbl, buf, mid0, tramp, selt, true);
    b.CreateBr(done);

    b.SetInsertPoint(c1);
    Value* mid1 = EmitLoadStoreSlot(b, tbl, buf, startoff, tramp, selt, true);
    (void)EmitLoadStoreSlot(b, tbl, buf, mid1, callee, selc, false);
    b.CreateBr(done);

    b.SetInsertPoint(done);
    return AddOff(b, startoff, 29);
}

static RuntimeRefs EmitRuntimeSkeleton(Module& m)
{
    LLVMContext& c = m.getContext();
    IntegerType* i32 = Type::getInt32Ty(c);
    Type* i64 = Type::getInt64Ty(c);
    PointerType* ptrty = PointerType::get(c, 0);

    RuntimeRefs r{};

    r.CacheEntryTy = StructType::create(c, {i64, ptrty}, "JITROPCacheEntry");
    r.CacheArrTy = ArrayType::get(r.CacheEntryTy, CacheSlots);

    auto mki64 = [&](const char* nm)
    {
        auto* gv = new GlobalVariable(m, i64, false, GlobalValue::InternalLinkage,
            ConstantInt::get(i64, 0), nm);
        gv->setAlignment(Align(8));
        return gv;
    };
    r.GvBareRet = mki64("JITROPBareRet");
    r.GvTramp40 = mki64("JITROPTramp40");
    r.GvTramp56 = mki64("JITROPTramp56");
    r.GvTramp72 = mki64("JITROPTramp72");
    r.GvTramp88 = mki64("JITROPTramp88");

    auto mkcache = [&](const char* nm)
    {
        auto* gv = new GlobalVariable(m, r.CacheArrTy, false, GlobalValue::InternalLinkage,
            ConstantAggregateZero::get(r.CacheArrTy), nm);
        gv->setAlignment(Align(16));
        return gv;
    };
    r.GvCacheBasic = mkcache("JITROPCacheBasic");
    r.GvCacheShadow = mkcache("JITROPCacheShadow");
    r.GvCacheCopy = mkcache("JITROPCacheCopy");
    r.GvCacheLoad = mkcache("JITROPCacheLoad");
    r.GvCacheStore = mkcache("JITROPCacheStore");
    r.GvCacheLea = mkcache("JITROPCacheLea");

    Constant* nts = ConstantDataArray::getString(c, "ntdll.dll", true);
    r.GvNtdllName = new GlobalVariable(m, nts->getType(), true,
        GlobalValue::PrivateLinkage, nts, "JITROPNtdllName");
    r.GvNtdllName->setUnnamedAddr(GlobalValue::UnnamedAddr::Global);

    uint8_t regraw[24] =
    {
        0x48, 0xB8,  0x48,  0x04,  0x44,  0xFF, 0xE0, 0x00,  // rax: mov rax,imm64 (48 B8) | store REX (48) | [rsp] ModRM (04) | [rsp+8] ModRM (44) | jmp rax (FF E0) | pad
        0x49, 0xBA,  0x4C,  0x14,  0x54,  0x41, 0xFF, 0xE2,  // r10: mov r10,imm64 (49 BA) | store REX (4C) | [rsp] ModRM (14) | [rsp+8] ModRM (54) | jmp r10 (41 FF E2)
        0x49, 0xBB,  0x4C,  0x1C,  0x5C,  0x41, 0xFF, 0xE3,  // r11: mov r11,imm64 (49 BB) | store REX (4C) | [rsp] ModRM (1C) | [rsp+8] ModRM (5C) | jmp r11 (41 FF E3)
    };
    Constant* ri = ConstantDataArray::get(c, ArrayRef<uint8_t>(regraw, 24));
    r.GvRegBytes = new GlobalVariable(m, ri->getType(), true,
        GlobalValue::PrivateLinkage, ri, "JITROPRegBytes");
    r.GvRegBytes->setUnnamedAddr(GlobalValue::UnnamedAddr::Global);
    r.GvRegBytes->setAlignment(Align(8));

    uint8_t saveraw[12] =
    {
        0x49, 0x89, 0xE2,  // mov r10, rsp  (89 form: REX.WB + 89 + ModRM D4->E2)
        0x49, 0x89, 0xE3,  // mov r11, rsp  (89 form: REX.WB + 89 + ModRM E3)
        0x4C, 0x8B, 0xD4,  // mov r10, rsp  (8B form: REX.WR + 8B + ModRM D4)
        0x4C, 0x8B, 0xDC,  // mov r11, rsp  (8B form: REX.WR + 8B + ModRM DC)
    };
    Constant* si = ConstantDataArray::get(c, ArrayRef<uint8_t>(saveraw, 12));
    r.GvSaveBytes = new GlobalVariable(m, si->getType(), true,
        GlobalValue::PrivateLinkage, si, "JITROPSaveBytes");
    r.GvSaveBytes->setUnnamedAddr(GlobalValue::UnnamedAddr::Global);
    r.GvSaveBytes->setAlignment(Align(4));

    uint8_t copyraw[16] =
    {
        0x49, 0x42, 0x48, 0x44,  // base=r10, xfer=rax : mov rax,[r10+d] loadREX=49 loadModRM=42 | mov [rsp+d'],rax storeREX=48 storeModRM=44
        0x4D, 0x5A, 0x4C, 0x5C,  // base=r10, xfer=r11 : mov r11,[r10+d] loadREX=4D loadModRM=5A | mov [rsp+d'],r11 storeREX=4C storeModRM=5C
        0x49, 0x43, 0x48, 0x44,  // base=r11, xfer=rax : mov rax,[r11+d] loadREX=49 loadModRM=43 | mov [rsp+d'],rax storeREX=48 storeModRM=44
        0x4D, 0x53, 0x4C, 0x54,  // base=r11, xfer=r10 : mov r10,[r11+d] loadREX=4D loadModRM=53 | mov [rsp+d'],r10 storeREX=4C storeModRM=54
    };
    Constant* ci2 = ConstantDataArray::get(c, ArrayRef<uint8_t>(copyraw, 16));
    r.GvCopyBytes = new GlobalVariable(m, ci2->getType(), true,
        GlobalValue::PrivateLinkage, ci2, "JITROPCopyBytes");
    r.GvCopyBytes->setUnnamedAddr(GlobalValue::UnnamedAddr::Global);
    r.GvCopyBytes->setAlignment(Align(4));

    auto keys = ComputeKeys(m);
    r.K1 = keys.first;
    r.K2 = keys.second;

    FunctionType* ftgmh = FunctionType::get(ptrty, {ptrty}, false);
    r.FnGMH = m.getOrInsertFunction("GetModuleHandleA", ftgmh);
    if (auto* f = dyn_cast<Function>(r.FnGMH.getCallee()))
        f->setDLLStorageClass(GlobalValue::DLLImportStorageClass);

    FunctionType* ftva = FunctionType::get(ptrty, {ptrty, i64, i32, i32}, false);
    r.FnVAlloc = m.getOrInsertFunction("VirtualAlloc", ftva);
    if (auto* f = dyn_cast<Function>(r.FnVAlloc.getCallee()))
        f->setDLLStorageClass(GlobalValue::DLLImportStorageClass);

    FunctionType* ftscan = FunctionType::get(i64, {}, false);
    FunctionType* ftbc2 = FunctionType::get(ptrty, {ptrty, i64}, false);
    FunctionType* ftbc3 = FunctionType::get(ptrty, {ptrty, i32, i64}, false);
    FunctionType* ftgc2 = FunctionType::get(ptrty, {i64, ptrty}, false);
    FunctionType* ftgc3 = FunctionType::get(ptrty, {i64, ptrty, i32}, false);
    FunctionType* ftbg3 = FunctionType::get(ptrty, {i64, i64, i32}, false);
    FunctionType* ftbg2 = FunctionType::get(ptrty, {i64, i64}, false);
    FunctionType* ftgg3 = FunctionType::get(ptrty, {i64, ptrty, i32}, false);
    FunctionType* ftgg2 = FunctionType::get(ptrty, {i64, ptrty}, false);

    auto mk = [&](FunctionType* ft, const char* nm)
    {
        Function* f = Function::Create(ft, GlobalValue::InternalLinkage, nm, &m);
        f->addFnAttr(Attribute::NoInline);
        f->addFnAttr(Attribute::NoUnwind);
        return f;
    };
    r.FnFindBareRet = mk(ftscan, "JITROPFindBareRet");
    r.FnFindTramp40 = mk(ftscan, "JITROPFindTramp40");
    r.FnFindTramp56 = mk(ftscan, "JITROPFindTramp56");
    r.FnFindTramp72 = mk(ftscan, "JITROPFindTramp72");
    r.FnFindTramp88 = mk(ftscan, "JITROPFindTramp88");
    r.FnBuildChainBasic = mk(ftbc2, "JITROPBuildChainBasic");
    r.FnBuildChainShadow = mk(ftbc2, "JITROPBuildChainShadow");
    r.FnBuildChainCopy = mk(ftbc3, "JITROPBuildChainCopy");
    r.FnGetChainBasic = mk(ftgc2, "JITROPGetChainBasic");
    r.FnGetChainShadow = mk(ftgc2, "JITROPGetChainShadow");
    r.FnGetChainCopy = mk(ftgc3, "JITROPGetChainCopy");
    r.FnBuildLoadGadget = mk(ftbg3, "JITROPBuildLoadGadget");
    r.FnBuildStoreGadget = mk(ftbg3, "JITROPBuildStoreGadget");
    r.FnBuildLeaGadget = mk(ftbg2, "JITROPBuildLeaGadget");
    r.FnGetLoadGadget = mk(ftgg3, "JITROPGetLoadGadget");
    r.FnGetStoreGadget = mk(ftgg3, "JITROPGetStoreGadget");
    r.FnGetLeaGadget = mk(ftgg2, "JITROPGetLeaGadget");

    return r;
}

static void EmitGadgetScanBody(Module& m, Function* f, GlobalVariable* cache, uint8_t pat, unsigned retoff, RuntimeRefs& r)
{
    LLVMContext& c = m.getContext();
    Type* i8 = Type::getInt8Ty(c);
    Type* i16 = Type::getInt16Ty(c);
    IntegerType* i32 = Type::getInt32Ty(c);
    Type* i64 = Type::getInt64Ty(c);
    PointerType* ptrty = PointerType::get(c, 0);
    Constant* zero64 = ConstantInt::get(i64, 0);
    Constant* nullp = ConstantPointerNull::get(ptrty);

    uint32_t pdword = ((uint32_t)pat << 24) | 0x00C48348U;

    BasicBlock* entry = BasicBlock::Create(c, "Entry", f);
    BasicBlock* havecached = BasicBlock::Create(c, "HaveCached", f);
    BasicBlock* compute = BasicBlock::Create(c, "Compute", f);
    BasicBlock* parsepe = BasicBlock::Create(c, "ParsePE", f);
    BasicBlock* secloop = BasicBlock::Create(c, "SecLoop", f);
    BasicBlock* secbody = BasicBlock::Create(c, "SecBody", f);
    BasicBlock* secgot = BasicBlock::Create(c, "SecGot", f);
    BasicBlock* secnext = BasicBlock::Create(c, "SecNext", f);
    BasicBlock* scanloop = BasicBlock::Create(c, "ScanLoop", f);
    BasicBlock* scanbody = BasicBlock::Create(c, "ScanBody", f);
    BasicBlock* scannext = BasicBlock::Create(c, "ScanNext", f);
    BasicBlock* found = BasicBlock::Create(c, "Found", f);
    BasicBlock* failret = BasicBlock::Create(c, "FailRet", f);

    IRBuilder<> b(entry);
    Value* cached = b.CreateAlignedLoad(i64, cache, Align(8), "Cached");
    b.CreateCondBr(b.CreateICmpNE(cached, zero64), havecached, compute);

    b.SetInsertPoint(havecached);
    b.CreateRet(cached);

    b.SetInsertPoint(compute);
    Value* base = b.CreateCall(r.FnGMH, {r.GvNtdllName}, "Base");
    b.CreateCondBr(b.CreateICmpEQ(base, nullp), failret, parsepe);

    b.SetInsertPoint(parsepe);
    Value* elfp = b.CreateGEP(i8, base, ConstantInt::get(i64, 0x3C));
    Value* elf32 = b.CreateAlignedLoad(i32, elfp, Align(1), "ELfanew");
    Value* elf64 = b.CreateZExt(elf32, i64);
    Value* nthdr = b.CreateGEP(i8, base, elf64, "NtHdr");

    Value* nsp = b.CreateGEP(i8, nthdr, ConstantInt::get(i64, 6));
    Value* ns16 = b.CreateAlignedLoad(i16, nsp, Align(1), "NumSecs");
    Value* ns32 = b.CreateZExt(ns16, i32);

    Value* osp = b.CreateGEP(i8, nthdr, ConstantInt::get(i64, 20));
    Value* os16 = b.CreateAlignedLoad(i16, osp, Align(1), "OptSz");
    Value* os64 = b.CreateZExt(os16, i64);

    Value* firstsec = b.CreateGEP(i8, nthdr,
        b.CreateAdd(ConstantInt::get(i64, 24), os64), "FirstSec");
    b.CreateBr(secloop);

    b.SetInsertPoint(secloop);
    PHINode* cursec = b.CreatePHI(ptrty, 2, "CurSec");
    PHINode* remain = b.CreatePHI(i32, 2, "Remain");
    cursec->addIncoming(firstsec, parsepe);
    remain->addIncoming(ns32, parsepe);
    b.CreateCondBr(b.CreateICmpUGT(remain, ConstantInt::get(i32, 0)), secbody, failret);

    b.SetInsertPoint(secbody);
    Value* nameld = b.CreateAlignedLoad(i64, cursec, Align(1), "NameLd");
    uint64_t textpat = (uint64_t)'.'
                     | ((uint64_t)'t' << 8)
                     | ((uint64_t)'e' << 16)
                     | ((uint64_t)'x' << 24)
                     | ((uint64_t)'t' << 32);
    Value* masked = b.CreateAnd(nameld, ConstantInt::get(i64, 0xFFFFFFFFFFULL));
    b.CreateCondBr(b.CreateICmpEQ(masked, ConstantInt::get(i64, textpat)), secgot, secnext);

    b.SetInsertPoint(secnext);
    Value* nextsec = b.CreateGEP(i8, cursec, ConstantInt::get(i64, 40));
    Value* remdec = b.CreateSub(remain, ConstantInt::get(i32, 1));
    cursec->addIncoming(nextsec, secnext);
    remain->addIncoming(remdec, secnext);
    b.CreateBr(secloop);

    b.SetInsertPoint(secgot);
    Value* vsp = b.CreateGEP(i8, cursec, ConstantInt::get(i64, 8));
    Value* vs32 = b.CreateAlignedLoad(i32, vsp, Align(1), "VSize");
    Value* vs64 = b.CreateZExt(vs32, i64);
    Value* vap = b.CreateGEP(i8, cursec, ConstantInt::get(i64, 12));
    Value* va32 = b.CreateAlignedLoad(i32, vap, Align(1), "VAddr");
    Value* va64 = b.CreateZExt(va32, i64);
    Value* secbase = b.CreateGEP(i8, base, va64, "SecBase");
    b.CreateCondBr(b.CreateICmpULT(vs64, ConstantInt::get(i64, 5)), failret, scanloop);

    b.SetInsertPoint(scanloop);
    PHINode* idx = b.CreatePHI(i64, 2, "Idx");
    idx->addIncoming(ConstantInt::get(i64, 0), secgot);
    Value* endidx = b.CreateSub(vs64, ConstantInt::get(i64, 4));
    b.CreateCondBr(b.CreateICmpUGE(idx, endidx), failret, scanbody);

    b.SetInsertPoint(scanbody);
    Value* p0 = b.CreateGEP(i8, secbase, idx);
    Value* v32 = b.CreateAlignedLoad(i32, p0, Align(1), "V32");
    Value* ca = b.CreateICmpEQ(v32, ConstantInt::get(i32, pdword));
    Value* p4 = b.CreateGEP(i8, secbase, b.CreateAdd(idx, ConstantInt::get(i64, 4)));
    Value* b4 = b.CreateAlignedLoad(i8, p4, Align(1), "B4");
    Value* cb = b.CreateICmpEQ(b4, ConstantInt::get(i8, 0xC3));
    b.CreateCondBr(b.CreateAnd(ca, cb), found, scannext);

    b.SetInsertPoint(scannext);
    Value* nextidx = b.CreateAdd(idx, ConstantInt::get(i64, 1));
    idx->addIncoming(nextidx, scannext);
    b.CreateBr(scanloop);

    b.SetInsertPoint(found);
    Value* gp = b.CreateGEP(i8, secbase, b.CreateAdd(idx, ConstantInt::get(i64, retoff)));
    Value* gi = b.CreatePtrToInt(gp, i64, "Gadget");
    b.CreateAlignedStore(gi, cache, Align(8));
    b.CreateRet(gi);

    b.SetInsertPoint(failret);
    b.CreateRet(zero64);
}

static void EmitBuildChainBasicBody(Module& m, RuntimeRefs& r)
{
    LLVMContext& c = m.getContext();
    IntegerType* i32 = Type::getInt32Ty(c);
    Type* i64 = Type::getInt64Ty(c);
    PointerType* ptrty = PointerType::get(c, 0);
    Constant* nullp = ConstantPointerNull::get(ptrty);

    Function* f = r.FnBuildChainBasic;
    auto ai = f->arg_begin();
    Value* callee = &*ai++; callee->setName("Callee");
    Value* sid = &*ai; sid->setName("Sid");

    BasicBlock* entry = BasicBlock::Create(c, "Entry", f);
    BasicBlock* havegadget = BasicBlock::Create(c, "HaveGadget", f);
    BasicBlock* havebuf = BasicBlock::Create(c, "HaveBuf", f);
    BasicBlock* failret = BasicBlock::Create(c, "FailRet", f);

    IRBuilder<> b(entry);
    Value* bareret = b.CreateCall(r.FnFindBareRet, {}, "BareRet");
    b.CreateCondBr(b.CreateICmpEQ(bareret, ConstantInt::get(i64, 0)), failret, havegadget);

    b.SetInsertPoint(havegadget);
    Value* buf = b.CreateCall(r.FnVAlloc,
        {nullp, ConstantInt::get(i64, 48),
         ConstantInt::get(i32, 0x3000), ConstantInt::get(i32, 0x40)}, "Buf");
    b.CreateCondBr(b.CreateICmpEQ(buf, nullp), failret, havebuf);

    b.SetInsertPoint(havebuf);
    ChainSelectors sel = ComputeChainSelectors(b, sid, i64);
    Value* off = EmitSubRSP(b, m, buf, ConstantInt::get(i64, 0), sel.Sub, ConstantInt::get(i64, 8));
    off = EmitLoadStoreSlot(b, r.GvRegBytes, buf, off, b.CreatePtrToInt(callee, i64), sel.C, false);
    (void)EmitLoadJmpSlot(b, r.GvRegBytes, buf, off, bareret, sel.B);
    b.CreateRet(buf);

    b.SetInsertPoint(failret);
    b.CreateRet(nullp);
}

static void EmitBuildChainShadowBody(Module& m, RuntimeRefs& r)
{
    LLVMContext& c = m.getContext();
    IntegerType* i32 = Type::getInt32Ty(c);
    Type* i64 = Type::getInt64Ty(c);
    PointerType* ptrty = PointerType::get(c, 0);
    Constant* nullp = ConstantPointerNull::get(ptrty);

    Function* f = r.FnBuildChainShadow;
    auto ai = f->arg_begin();
    Value* callee = &*ai++; callee->setName("Callee");
    Value* sid = &*ai; sid->setName("Sid");

    BasicBlock* entry = BasicBlock::Create(c, "Entry", f);
    BasicBlock* havebare = BasicBlock::Create(c, "HaveBare", f);
    BasicBlock* haveboth = BasicBlock::Create(c, "HaveBoth", f);
    BasicBlock* havebuf = BasicBlock::Create(c, "HaveBuf", f);
    BasicBlock* failret = BasicBlock::Create(c, "FailRet", f);

    IRBuilder<> b(entry);
    Value* bareret = b.CreateCall(r.FnFindBareRet, {}, "BareRet");
    b.CreateCondBr(b.CreateICmpEQ(bareret, ConstantInt::get(i64, 0)), failret, havebare);

    b.SetInsertPoint(havebare);
    Value* tramp = b.CreateCall(r.FnFindTramp40, {}, "Tramp40");
    b.CreateCondBr(b.CreateICmpEQ(tramp, ConstantInt::get(i64, 0)), failret, haveboth);

    b.SetInsertPoint(haveboth);
    Value* buf = b.CreateCall(r.FnVAlloc,
        {nullp, ConstantInt::get(i64, 64),
         ConstantInt::get(i32, 0x3000), ConstantInt::get(i32, 0x40)}, "Buf");
    b.CreateCondBr(b.CreateICmpEQ(buf, nullp), failret, havebuf);

    b.SetInsertPoint(havebuf);
    ChainSelectors sel = ComputeChainSelectors(b, sid, i64);
    Value* off = EmitSubRSP(b, m, buf, ConstantInt::get(i64, 0), sel.Sub, ConstantInt::get(i64, 56));
    off = EmitTailPair(b, r.GvRegBytes, buf, off,
        b.CreatePtrToInt(callee, i64), sel.C, tramp, sel.T, sel.Order);
    (void)EmitLoadJmpSlot(b, r.GvRegBytes, buf, off, bareret, sel.B);
    b.CreateRet(buf);

    b.SetInsertPoint(failret);
    b.CreateRet(nullp);
}

static void EmitBuildChainCopyBody(Module& m, RuntimeRefs& r)
{
    LLVMContext& c = m.getContext();
    Type* i8 = Type::getInt8Ty(c);
    IntegerType* i32 = Type::getInt32Ty(c);
    Type* i64 = Type::getInt64Ty(c);
    PointerType* ptrty = PointerType::get(c, 0);
    Constant* nullp = ConstantPointerNull::get(ptrty);

    Function* f = r.FnBuildChainCopy;
    auto ai = f->arg_begin();
    Value* callee = &*ai++; callee->setName("Callee");
    Value* nsa = &*ai++; nsa->setName("NStackArgs");
    Value* sid = &*ai; sid->setName("Sid");

    BasicBlock* entry = BasicBlock::Create(c, "Entry", f);
    BasicBlock* checkbare = BasicBlock::Create(c, "CheckBare", f);
    BasicBlock* dispatch = BasicBlock::Create(c, "Dispatch", f);
    BasicBlock* cmpb = BasicBlock::Create(c, "CmpB", f);
    BasicBlock* bucketa = BasicBlock::Create(c, "BucketA", f);
    BasicBlock* bucketb = BasicBlock::Create(c, "BucketB", f);
    BasicBlock* bucketc = BasicBlock::Create(c, "BucketC", f);
    BasicBlock* havetramp = BasicBlock::Create(c, "HaveTramp", f);
    BasicBlock* haveboth = BasicBlock::Create(c, "HaveBoth", f);
    BasicBlock* havebuf = BasicBlock::Create(c, "HaveBuf", f);
    BasicBlock* copyloop = BasicBlock::Create(c, "CopyLoop", f);
    BasicBlock* copybody = BasicBlock::Create(c, "CopyBody", f);
    BasicBlock* copydone = BasicBlock::Create(c, "CopyDone", f);
    BasicBlock* failret = BasicBlock::Create(c, "FailRet", f);

    IRBuilder<> b(entry);
    Value* toobig = b.CreateICmpUGT(nsa, ConstantInt::get(i32, 7));
    b.CreateCondBr(toobig, failret, checkbare);

    b.SetInsertPoint(checkbare);
    Value* bareret = b.CreateCall(r.FnFindBareRet, {}, "BareRet");
    b.CreateCondBr(b.CreateICmpEQ(bareret, ConstantInt::get(i64, 0)), failret, dispatch);

    b.SetInsertPoint(dispatch);
    Value* leq3 = b.CreateICmpULE(nsa, ConstantInt::get(i32, 3));
    b.CreateCondBr(leq3, bucketa, cmpb);

    b.SetInsertPoint(cmpb);
    Value* leq5 = b.CreateICmpULE(nsa, ConstantInt::get(i32, 5));
    b.CreateCondBr(leq5, bucketb, bucketc);

    b.SetInsertPoint(bucketa);
    Value* tra = b.CreateCall(r.FnFindTramp56, {}, "TrampA");
    b.CreateBr(havetramp);

    b.SetInsertPoint(bucketb);
    Value* trb = b.CreateCall(r.FnFindTramp72, {}, "TrampB");
    b.CreateBr(havetramp);

    b.SetInsertPoint(bucketc);
    Value* trc = b.CreateCall(r.FnFindTramp88, {}, "TrampC");
    b.CreateBr(havetramp);

    b.SetInsertPoint(havetramp);
    PHINode* tramp = b.CreatePHI(i64, 3, "Tramp");
    tramp->addIncoming(tra, bucketa);
    tramp->addIncoming(trb, bucketb);
    tramp->addIncoming(trc, bucketc);
    PHINode* subimm = b.CreatePHI(i64, 3, "SubImm");
    subimm->addIncoming(ConstantInt::get(i64, 72), bucketa);
    subimm->addIncoming(ConstantInt::get(i64, 88), bucketb);
    subimm->addIncoming(ConstantInt::get(i64, 104), bucketc);
    b.CreateCondBr(b.CreateICmpEQ(tramp, ConstantInt::get(i64, 0)), failret, haveboth);

    b.SetInsertPoint(haveboth);
    Value* buf = b.CreateCall(r.FnVAlloc,
        {nullp, ConstantInt::get(i64, 128),
         ConstantInt::get(i32, 0x3000), ConstantInt::get(i32, 0x40)}, "Buf");
    b.CreateCondBr(b.CreateICmpEQ(buf, nullp), failret, havebuf);

    b.SetInsertPoint(havebuf);
    ChainSelectors sel = ComputeChainSelectors(b, sid, i64);
    Value* saveend = EmitSaveMov(b, r.GvSaveBytes, buf, ConstantInt::get(i64, 0), sel.Save, sel.SaveReg);
    Value* subend = EmitSubRSP(b, m, buf, saveend, sel.Sub, subimm);
    Value* copystart = subend;

    CopyBytes cb = ReadCopyBytes(b, r.GvCopyBytes, sel.SaveReg, sel.Xfer);

    Value* nsa64 = b.CreateZExt(nsa, i64);
    // capture the current BB AFTER emitSubRSP returns
    BasicBlock* preloop = b.GetInsertBlock();
    b.CreateBr(copyloop);

    b.SetInsertPoint(copyloop);
    PHINode* iter = b.CreatePHI(i64, 2, "I");
    iter->addIncoming(ConstantInt::get(i64, 0), preloop);
    b.CreateCondBr(b.CreateICmpULT(iter, nsa64), copybody, copydone);

    b.SetInsertPoint(copybody);
    Value* im9 = b.CreateMul(iter, ConstantInt::get(i64, 9));
    Value* basei = b.CreateAdd(copystart, im9);
    Value* im8 = b.CreateMul(iter, ConstantInt::get(i64, 8));
    Value* dispsrc = b.CreateTrunc(b.CreateAdd(ConstantInt::get(i64, 40), im8), i8);
    Value* dispdst = b.CreateTrunc(b.CreateAdd(ConstantInt::get(i64, 48), im8), i8);

    auto wca = [&](Value* off, uint8_t v)
    {
        Value* p = b.CreateGEP(i8, buf, off);
        b.CreateAlignedStore(ConstantInt::get(i8, v), p, Align(1));
    };
    auto wva = [&](Value* off, Value* v)
    {
        Value* p = b.CreateGEP(i8, buf, off);
        b.CreateAlignedStore(v, p, Align(1));
    };
    // mov <xfer>, [<base> + 40+8n]    REX + 8B + ModRM + disp8
    wva(basei, cb.LoadREX);                                              // load REX
    wca(b.CreateAdd(basei, ConstantInt::get(i64, 1)), 0x8B);             // MOV r64, r/m64 opcode
    wva(b.CreateAdd(basei, ConstantInt::get(i64, 2)), cb.LoadModRM);     // load ModRM (xfer, [base+disp8])
    wva(b.CreateAdd(basei, ConstantInt::get(i64, 3)), dispsrc);          // disp8 = 40 + 8n
    // mov [rsp + 48+8n], <xfer>       REX + 89 + ModRM + SIB(0x24) + disp8
    wva(b.CreateAdd(basei, ConstantInt::get(i64, 4)), cb.StoreREX);      // store REX
    wca(b.CreateAdd(basei, ConstantInt::get(i64, 5)), 0x89);             // MOV r/m64, r64 opcode
    wva(b.CreateAdd(basei, ConstantInt::get(i64, 6)), cb.StoreModRM);    // store ModRM ([rsp+disp8], xfer)
    wca(b.CreateAdd(basei, ConstantInt::get(i64, 7)), 0x24);             // SIB base=rsp
    wva(b.CreateAdd(basei, ConstantInt::get(i64, 8)), dispdst);          // disp8 = 48 + 8n

    iter->addIncoming(b.CreateAdd(iter, ConstantInt::get(i64, 1)), copybody);
    b.CreateBr(copyloop);

    b.SetInsertPoint(copydone);
    Value* tailbase = b.CreateAdd(copystart, b.CreateMul(nsa64, ConstantInt::get(i64, 9)));
    Value* off = EmitTailPair(b, r.GvRegBytes, buf, tailbase,
        b.CreatePtrToInt(callee, i64), sel.C, tramp, sel.T, sel.Order);
    (void)EmitLoadJmpSlot(b, r.GvRegBytes, buf, off, bareret, sel.B);
    b.CreateRet(buf);

    b.SetInsertPoint(failret);
    b.CreateRet(nullp);
}

static void EmitBuildLoadGadgetBody(Module& m, RuntimeRefs& r)
{
    LLVMContext& c = m.getContext();
    Type* i8 = Type::getInt8Ty(c);
    IntegerType* i32 = Type::getInt32Ty(c);
    Type* i64 = Type::getInt64Ty(c);
    PointerType* ptrty = PointerType::get(c, 0);
    Constant* nullp = ConstantPointerNull::get(ptrty);

    Function* f = r.FnBuildLoadGadget;
    auto ai = f->arg_begin();
    Value* addr = &*ai++; addr->setName("Addr");
    Value* sid = &*ai++; sid->setName("Sid"); (void)sid;
    Value* size = &*ai; size->setName("Size");

    BasicBlock* entry = BasicBlock::Create(c, "Entry", f);
    BasicBlock* havebuf = BasicBlock::Create(c, "HaveBuf", f);
    BasicBlock* s8 = BasicBlock::Create(c, "S8", f);
    BasicBlock* s4 = BasicBlock::Create(c, "S4", f);
    BasicBlock* s2 = BasicBlock::Create(c, "S2", f);
    BasicBlock* s1 = BasicBlock::Create(c, "S1", f);
    BasicBlock* done = BasicBlock::Create(c, "Done", f);
    BasicBlock* fail = BasicBlock::Create(c, "Fail", f);

    IRBuilder<> b(entry);
    Value* buf = b.CreateCall(r.FnVAlloc,
        {nullp, ConstantInt::get(i64, GadgetBufSize),
         ConstantInt::get(i32, 0x3000), ConstantInt::get(i32, 0x40)}, "Buf");
    b.CreateCondBr(b.CreateICmpEQ(buf, nullp), fail, havebuf);

    b.SetInsertPoint(havebuf);
    auto wc = [&](uint64_t off, uint8_t v)
    {
        Value* p = b.CreateGEP(i8, buf, ConstantInt::get(i64, off));
        b.CreateAlignedStore(ConstantInt::get(i8, v), p, Align(1));
    };
    auto w64 = [&](uint64_t off, Value* v)
    {
        Value* p = b.CreateGEP(i8, buf, ConstantInt::get(i64, off));
        b.CreateAlignedStore(v, p, Align(1));
    };
    // common prefix: mov rax, addr   48 B8 <imm64>
    wc(0, 0x48);
    wc(1, 0xB8);
    w64(2, addr);
    SwitchInst* sw = b.CreateSwitch(size, s8, 4);
    sw->addCase(ConstantInt::get(i32, 1), s1);
    sw->addCase(ConstantInt::get(i32, 2), s2);
    sw->addCase(ConstantInt::get(i32, 4), s4);
    sw->addCase(ConstantInt::get(i32, 8), s8);

    b.SetInsertPoint(s8);
    // mov rax, [rax]  ;  ret       48 8B 00 C3
    wc(10, 0x48); wc(11, 0x8B); wc(12, 0x00); wc(13, 0xC3);
    b.CreateBr(done);

    b.SetInsertPoint(s4);
    // mov eax, [rax]  ;  ret       8B 00 C3   (eax zero-extends to rax)
    wc(10, 0x8B); wc(11, 0x00); wc(12, 0xC3);
    b.CreateBr(done);

    b.SetInsertPoint(s2);
    // movzx eax, word [rax] ; ret  0F B7 00 C3
    wc(10, 0x0F); wc(11, 0xB7); wc(12, 0x00); wc(13, 0xC3);
    b.CreateBr(done);

    b.SetInsertPoint(s1);
    // movzx eax, byte [rax] ; ret  0F B6 00 C3
    wc(10, 0x0F); wc(11, 0xB6); wc(12, 0x00); wc(13, 0xC3);
    b.CreateBr(done);

    b.SetInsertPoint(done);
    b.CreateRet(buf);

    b.SetInsertPoint(fail);
    b.CreateRet(nullp);
}

static void EmitBuildStoreGadgetBody(Module& m, RuntimeRefs& r)
{
    LLVMContext& c = m.getContext();
    Type* i8 = Type::getInt8Ty(c);
    IntegerType* i32 = Type::getInt32Ty(c);
    Type* i64 = Type::getInt64Ty(c);
    PointerType* ptrty = PointerType::get(c, 0);
    Constant* nullp = ConstantPointerNull::get(ptrty);

    Function* f = r.FnBuildStoreGadget;
    auto ai = f->arg_begin();
    Value* addr = &*ai++; addr->setName("Addr");
    Value* sid = &*ai++; sid->setName("Sid"); (void)sid;
    Value* size = &*ai; size->setName("Size");

    BasicBlock* entry = BasicBlock::Create(c, "Entry", f);
    BasicBlock* havebuf = BasicBlock::Create(c, "HaveBuf", f);
    BasicBlock* s8 = BasicBlock::Create(c, "S8", f);
    BasicBlock* s4 = BasicBlock::Create(c, "S4", f);
    BasicBlock* s2 = BasicBlock::Create(c, "S2", f);
    BasicBlock* s1 = BasicBlock::Create(c, "S1", f);
    BasicBlock* done = BasicBlock::Create(c, "Done", f);
    BasicBlock* fail = BasicBlock::Create(c, "Fail", f);

    IRBuilder<> b(entry);
    Value* buf = b.CreateCall(r.FnVAlloc,
        {nullp, ConstantInt::get(i64, GadgetBufSize),
         ConstantInt::get(i32, 0x3000), ConstantInt::get(i32, 0x40)}, "Buf");
    b.CreateCondBr(b.CreateICmpEQ(buf, nullp), fail, havebuf);

    b.SetInsertPoint(havebuf);
    auto wc = [&](uint64_t off, uint8_t v)
    {
        Value* p = b.CreateGEP(i8, buf, ConstantInt::get(i64, off));
        b.CreateAlignedStore(ConstantInt::get(i8, v), p, Align(1));
    };
    auto w64 = [&](uint64_t off, Value* v)
    {
        Value* p = b.CreateGEP(i8, buf, ConstantInt::get(i64, off));
        b.CreateAlignedStore(v, p, Align(1));
    };
    // common prefix: mov rax, addr   48 B8 <imm64>
    wc(0, 0x48);
    wc(1, 0xB8);
    w64(2, addr);
    SwitchInst* sw = b.CreateSwitch(size, s8, 4);
    sw->addCase(ConstantInt::get(i32, 1), s1);
    sw->addCase(ConstantInt::get(i32, 2), s2);
    sw->addCase(ConstantInt::get(i32, 4), s4);
    sw->addCase(ConstantInt::get(i32, 8), s8);

    b.SetInsertPoint(s8);
    // mov [rax], rcx  ;  ret       48 89 08 C3
    wc(10, 0x48); wc(11, 0x89); wc(12, 0x08); wc(13, 0xC3);
    b.CreateBr(done);

    b.SetInsertPoint(s4);
    // mov [rax], ecx  ;  ret       89 08 C3
    wc(10, 0x89); wc(11, 0x08); wc(12, 0xC3);
    b.CreateBr(done);

    b.SetInsertPoint(s2);
    // mov [rax], cx   ;  ret       66 89 08 C3  (operand-size prefix)
    wc(10, 0x66); wc(11, 0x89); wc(12, 0x08); wc(13, 0xC3);
    b.CreateBr(done);

    b.SetInsertPoint(s1);
    // mov [rax], cl   ;  ret       88 08 C3
    wc(10, 0x88); wc(11, 0x08); wc(12, 0xC3);
    b.CreateBr(done);

    b.SetInsertPoint(done);
    b.CreateRet(buf);

    b.SetInsertPoint(fail);
    b.CreateRet(nullp);
}

static void EmitBuildLeaGadgetBody(Module& m, RuntimeRefs& r)
{
    LLVMContext& c = m.getContext();
    Type* i8 = Type::getInt8Ty(c);
    IntegerType* i32 = Type::getInt32Ty(c);
    Type* i64 = Type::getInt64Ty(c);
    PointerType* ptrty = PointerType::get(c, 0);
    Constant* nullp = ConstantPointerNull::get(ptrty);

    Function* f = r.FnBuildLeaGadget;
    auto ai = f->arg_begin();
    Value* addr = &*ai++; addr->setName("Addr");
    Value* sid = &*ai; sid->setName("Sid"); (void)sid;

    BasicBlock* entry = BasicBlock::Create(c, "Entry", f);
    BasicBlock* havebuf = BasicBlock::Create(c, "HaveBuf", f);
    BasicBlock* fail = BasicBlock::Create(c, "Fail", f);

    IRBuilder<> b(entry);
    Value* buf = b.CreateCall(r.FnVAlloc,
        {nullp, ConstantInt::get(i64, GadgetBufSize),
         ConstantInt::get(i32, 0x3000), ConstantInt::get(i32, 0x40)}, "Buf");
    b.CreateCondBr(b.CreateICmpEQ(buf, nullp), fail, havebuf);

    b.SetInsertPoint(havebuf);
    auto wc = [&](uint64_t off, uint8_t v)
    {
        Value* p = b.CreateGEP(i8, buf, ConstantInt::get(i64, off));
        b.CreateAlignedStore(ConstantInt::get(i8, v), p, Align(1));
    };
    auto w64 = [&](uint64_t off, Value* v)
    {
        Value* p = b.CreateGEP(i8, buf, ConstantInt::get(i64, off));
        b.CreateAlignedStore(v, p, Align(1));
    };
    // mov rax, addr  ;  ret        48 B8 <imm64> C3
    wc(0, 0x48);
    wc(1, 0xB8);
    w64(2, addr);
    wc(10, 0xC3);
    b.CreateRet(buf);

    b.SetInsertPoint(fail);
    b.CreateRet(nullp);
}

static void EmitGetChainBody(LLVMContext& c, Function* f, Function* buildfn, GlobalVariable* cache, StructType* etyp, ArrayType* atyp, uint64_t k2, bool hasnsa)
{
    Type* i64 = Type::getInt64Ty(c);
    PointerType* ptrty = PointerType::get(c, 0);
    Constant* nullp = ConstantPointerNull::get(ptrty);
    Constant* k2c = ConstantInt::get(i64, k2);

    auto ai = f->arg_begin();
    Value* sid = &*ai++; sid->setName("Sid");
    Value* slot = &*ai++; slot->setName("CalleeSlot");
    Value* nsa = hasnsa ? &*ai : nullptr;
    if (nsa) nsa->setName("NStackArgs");

    BasicBlock* entry = BasicBlock::Create(c, "Entry", f);
    BasicBlock* head = BasicBlock::Create(c, "LoopHead", f);
    BasicBlock* body = BasicBlock::Create(c, "LoopBody", f);
    BasicBlock* hit = BasicBlock::Create(c, "Hit", f);
    BasicBlock* checkemp = BasicBlock::Create(c, "CheckEmp", f);
    BasicBlock* build = BasicBlock::Create(c, "Build", f);
    BasicBlock* dostore = BasicBlock::Create(c, "DoStore", f);
    BasicBlock* retchain = BasicBlock::Create(c, "RetChain", f);
    BasicBlock* retcallee = BasicBlock::Create(c, "RetCallee", f);
    BasicBlock* next = BasicBlock::Create(c, "LoopNext", f);
    BasicBlock* full = BasicBlock::Create(c, "Full", f);

    Constant* slotmask = ConstantInt::get(i64, CacheSlots - 1);

    IRBuilder<> b(entry);
    Value* callee = b.CreateAlignedLoad(ptrty, slot, Align(8), "Callee");
    Value* start = b.CreateAnd(sid, slotmask);
    b.CreateBr(head);

    b.SetInsertPoint(head);
    PHINode* iter = b.CreatePHI(i64, 2, "Iter");
    iter->addIncoming(ConstantInt::get(i64, 0), entry);
    b.CreateCondBr(b.CreateICmpUGE(iter, ConstantInt::get(i64, CacheSlots)), full, body);

    b.SetInsertPoint(body);
    Value* idx = b.CreateAnd(b.CreateAdd(start, iter), slotmask);
    Value* ep = b.CreateGEP(atyp, cache, {ConstantInt::get(i64, 0), idx});
    Value* sidp = b.CreateStructGEP(etyp, ep, 0, "SidP");
    Value* chp = b.CreateStructGEP(etyp, ep, 1, "ChainP");

    LoadInst* cur = b.CreateAlignedLoad(i64, sidp, Align(8), "CurId");
    cur->setAtomic(AtomicOrdering::Acquire);
    b.CreateCondBr(b.CreateICmpEQ(cur, sid), hit, checkemp);

    b.SetInsertPoint(hit);
    Value* hitc = b.CreateAlignedLoad(ptrty, chp, Align(8), "HitChain");
    Value* hi = b.CreatePtrToInt(hitc, i64);
    Value* di = b.CreateXor(hi, k2c);
    Value* dp = b.CreateIntToPtr(di, ptrty, "DecChain");
    b.CreateRet(dp);

    b.SetInsertPoint(checkemp);
    b.CreateCondBr(b.CreateICmpEQ(cur, ConstantInt::get(i64, 0)), build, next);

    b.SetInsertPoint(build);
    Value* chain = nsa
        ? b.CreateCall(buildfn, {callee, nsa, sid}, "Chain")
        : b.CreateCall(buildfn, {callee, sid}, "Chain");
    b.CreateCondBr(b.CreateICmpEQ(chain, nullp), retcallee, dostore);

    b.SetInsertPoint(dostore);
    Value* ci = b.CreatePtrToInt(chain, i64);
    Value* ei = b.CreateXor(ci, k2c);
    Value* enp = b.CreateIntToPtr(ei, ptrty, "EncChain");
    b.CreateAlignedStore(enp, chp, Align(8));
    // store the encrypted chainptr BEFORE the release store of sid
    StoreInst* ss = b.CreateAlignedStore(sid, sidp, Align(8));
    ss->setAtomic(AtomicOrdering::Release);
    b.CreateBr(retchain);

    b.SetInsertPoint(retchain);
    b.CreateRet(chain);

    b.SetInsertPoint(retcallee);
    b.CreateRet(callee);

    b.SetInsertPoint(next);
    iter->addIncoming(b.CreateAdd(iter, ConstantInt::get(i64, 1)), next);
    b.CreateBr(head);

    b.SetInsertPoint(full);
    b.CreateRet(callee);
}

static void EmitGetGadgetBody(LLVMContext& c, Function* f, Function* buildfn, GlobalVariable* cache, StructType* etyp, ArrayType* atyp, uint64_t k2, bool hassize)
{
    Type* i64 = Type::getInt64Ty(c);
    PointerType* ptrty = PointerType::get(c, 0);
    Constant* nullp = ConstantPointerNull::get(ptrty);
    Constant* k2c = ConstantInt::get(i64, k2);

    auto ai = f->arg_begin();
    Value* sid = &*ai++; sid->setName("Sid");
    Value* slot = &*ai++; slot->setName("AddrSlot");
    Value* sizev = hassize ? &*ai : nullptr;
    if (sizev) sizev->setName("Size");

    BasicBlock* entry = BasicBlock::Create(c, "Entry", f);
    BasicBlock* head = BasicBlock::Create(c, "LoopHead", f);
    BasicBlock* body = BasicBlock::Create(c, "LoopBody", f);
    BasicBlock* hit = BasicBlock::Create(c, "Hit", f);
    BasicBlock* checkemp = BasicBlock::Create(c, "CheckEmp", f);
    BasicBlock* build = BasicBlock::Create(c, "Build", f);
    BasicBlock* dostore = BasicBlock::Create(c, "DoStore", f);
    BasicBlock* retgadget = BasicBlock::Create(c, "RetGadget", f);
    BasicBlock* retnull = BasicBlock::Create(c, "RetNull", f);
    BasicBlock* next = BasicBlock::Create(c, "LoopNext", f);
    BasicBlock* full = BasicBlock::Create(c, "Full", f);

    Constant* slotmask = ConstantInt::get(i64, CacheSlots - 1);

    IRBuilder<> b(entry);
    Value* realaddr = b.CreateAlignedLoad(ptrty, slot, Align(8), "RealAddr");
    Value* addri = b.CreatePtrToInt(realaddr, i64, "AddrI");
    Value* start = b.CreateAnd(sid, slotmask);
    b.CreateBr(head);

    b.SetInsertPoint(head);
    PHINode* iter = b.CreatePHI(i64, 2, "Iter");
    iter->addIncoming(ConstantInt::get(i64, 0), entry);
    b.CreateCondBr(b.CreateICmpUGE(iter, ConstantInt::get(i64, CacheSlots)), full, body);

    b.SetInsertPoint(body);
    Value* idx = b.CreateAnd(b.CreateAdd(start, iter), slotmask);
    Value* ep = b.CreateGEP(atyp, cache, {ConstantInt::get(i64, 0), idx});
    Value* sidp = b.CreateStructGEP(etyp, ep, 0, "SidP");
    Value* gp = b.CreateStructGEP(etyp, ep, 1, "GadgetP");

    LoadInst* cur = b.CreateAlignedLoad(i64, sidp, Align(8), "CurId");
    cur->setAtomic(AtomicOrdering::Acquire);
    b.CreateCondBr(b.CreateICmpEQ(cur, sid), hit, checkemp);

    b.SetInsertPoint(hit);
    Value* hitg = b.CreateAlignedLoad(ptrty, gp, Align(8), "HitEnc");
    Value* hi = b.CreatePtrToInt(hitg, i64);
    Value* di = b.CreateXor(hi, k2c);
    Value* dp = b.CreateIntToPtr(di, ptrty, "DecGadget");
    b.CreateRet(dp);

    b.SetInsertPoint(checkemp);
    b.CreateCondBr(b.CreateICmpEQ(cur, ConstantInt::get(i64, 0)), build, next);

    b.SetInsertPoint(build);
    Value* g = hassize
        ? b.CreateCall(buildfn, {addri, sid, sizev}, "Gadget")
        : b.CreateCall(buildfn, {addri, sid}, "Gadget");
    b.CreateCondBr(b.CreateICmpEQ(g, nullp), retnull, dostore);

    b.SetInsertPoint(dostore);
    Value* gi = b.CreatePtrToInt(g, i64);
    Value* ei = b.CreateXor(gi, k2c);
    Value* enp = b.CreateIntToPtr(ei, ptrty, "EncGadget");
    b.CreateAlignedStore(enp, gp, Align(8));
    StoreInst* ss = b.CreateAlignedStore(sid, sidp, Align(8));
    ss->setAtomic(AtomicOrdering::Release);
    b.CreateBr(retgadget);

    b.SetInsertPoint(retgadget);
    b.CreateRet(g);

    b.SetInsertPoint(retnull);
    b.CreateRet(nullp);

    b.SetInsertPoint(next);
    iter->addIncoming(b.CreateAdd(iter, ConstantInt::get(i64, 1)), next);
    b.CreateBr(head);

    b.SetInsertPoint(full);
    b.CreateRet(nullp);
}

static Value* BuildAddrSlotArg(IRBuilder<>& b, RuntimeRefs& r, GlobalVariable* target)
{
    Type* i64 = Type::getInt64Ty(b.getContext());
    PointerType* ptrty = PointerType::get(b.getContext(), 0);

    if (!r.GvAddrTable) return b.CreatePointerCast(target, ptrty);
    auto it = r.AddrIndex.find(target);
    if (it == r.AddrIndex.end()) return b.CreatePointerCast(target, ptrty);
    ArrayType* tyarr = cast<ArrayType>(r.GvAddrTable->getValueType());
    return b.CreateGEP(tyarr, r.GvAddrTable,
        {ConstantInt::get(i64, 0), ConstantInt::get(i64, it->second)}, "AddrSlot");
}

static void RewriteCallSite(CallInst* ci, uint64_t sid, unsigned idx, RuntimeRefs& r)
{
    LLVMContext& c = ci->getContext();
    IntegerType* i32 = Type::getInt32Ty(c);
    Type* i64 = Type::getInt64Ty(c);
    PointerType* ptrty = PointerType::get(c, 0);
    Constant* nullp = ConstantPointerNull::get(ptrty);

    const unsigned nargs = (unsigned)ci->arg_size();
    CallShape shape;
    if (nargs <= 4) shape = ShapeShadow;
    else if (nargs <= 11) shape = ShapeCopy;
    else shape = ShapeBasic;

    Function* getfn =
        shape == ShapeShadow ? r.FnGetChainShadow :
        shape == ShapeCopy ? r.FnGetChainCopy :
                             r.FnGetChainBasic;

    Function* directcallee = ci->getCalledFunction();
    IRBuilder<> b(ci);

    auto emitargs = [&](IRBuilder<>& bb) -> std::pair<Value*, Value*>
    {
        Value* sidc = ConstantInt::get(i64, SafeEncSid(sid, r.K1));
        Value* cp = nullptr;
        if (directcallee && r.GvCalleeTable)
        {
            auto it = r.CalleeIndex.find(directcallee);
            if (it != r.CalleeIndex.end())
            {
                unsigned slotidx = it->second;
                ArrayType* tyarr = cast<ArrayType>(r.GvCalleeTable->getValueType());
                cp = bb.CreateGEP(tyarr, r.GvCalleeTable,
                    {ConstantInt::get(i64, 0), ConstantInt::get(i64, slotidx)}, "CalleeSlot");
            }
            else
            {
                cp = bb.CreatePointerCast(directcallee, ptrty);
            }
        }
        else
        {
            cp = bb.CreatePointerCast(directcallee, ptrty);
        }
        return {sidc, cp};
    };

    auto emitcall = [&](IRBuilder<>& bb) -> Value*
    {
        auto [sidc, cp] = emitargs(bb);
        if (shape == ShapeCopy)
        {
            Value* nsac = ConstantInt::get(i32, nargs - 4);
            return bb.CreateCall(getfn, {sidc, cp, nsac}, "Chain");
        }
        return bb.CreateCall(getfn, {sidc, cp}, "Chain");
    };

    Value* chain = nullptr;
    if (r.GvSiteCache)
    {
        ArrayType* sitety = cast<ArrayType>(r.GvSiteCache->getValueType());
        Value* slot = b.CreateInBoundsGEP(sitety, r.GvSiteCache,
            {ConstantInt::get(i64, 0), ConstantInt::get(i64, idx)}, "SiteSlot");
        LoadInst* cached = b.CreateAlignedLoad(ptrty, slot, Align(8), "Cached");
        Value* isnull = b.CreateICmpEQ(cached, nullp, "IsMiss");

        BasicBlock* prebb = b.GetInsertBlock();
        BasicBlock* donebb = prebb->splitBasicBlock(ci, "Done");
        prebb->getTerminator()->eraseFromParent();

        BasicBlock* missbb = BasicBlock::Create(c, "Miss", prebb->getParent(), donebb);

        b.SetInsertPoint(prebb);
        b.CreateCondBr(isnull, missbb, donebb);

        b.SetInsertPoint(missbb);
        Value* fresh = emitcall(b);
        b.CreateAlignedStore(fresh, slot, Align(8));
        b.CreateBr(donebb);

        // CreatePHI with insert point at &donebb->front() places the PHI before `ci` (the first instruction after the split)
        b.SetInsertPoint(&donebb->front());
        PHINode* phi = b.CreatePHI(ptrty, 2, "Chain");
        phi->addIncoming(cached, prebb);
        phi->addIncoming(fresh, missbb);
        chain = phi;

        b.SetInsertPoint(ci);
    }
    else
    {
        chain = emitcall(b);
    }

    std::vector<Value*> args(ci->arg_begin(), ci->arg_end());
    CallInst* newcall = b.CreateCall(ci->getFunctionType(), chain, args);
    newcall->setCallingConv(ci->getCallingConv());
    newcall->setAttributes(ci->getAttributes());
    newcall->setDebugLoc(ci->getDebugLoc());
    newcall->setTailCallKind(CallInst::TCK_None);

    if (!ci->getType()->isVoidTy()) ci->replaceAllUsesWith(newcall);
    ci->eraseFromParent();
}

static void RewriteLoad(OpSite& si, RuntimeRefs& r)
{
    auto* li = cast<LoadInst>(si.Inst);
    LLVMContext& c = li->getContext();
    IntegerType* i32 = Type::getInt32Ty(c);
    Type* i64 = Type::getInt64Ty(c);
    PointerType* ptrty = PointerType::get(c, 0);

    IRBuilder<> b(li);
    Value* slotarg = BuildAddrSlotArg(b, r, si.Target);
    Value* sidc = ConstantInt::get(i64, SafeEncSid(si.Sid, r.K1));
    Value* sizec = ConstantInt::get(i32, si.SizeBytes);

    Value* gadget = b.CreateCall(r.FnGetLoadGadget, {sidc, slotarg, sizec}, "Gadget");
    FunctionType* gty = FunctionType::get(i64, {}, false);
    Value* raw = b.CreateCall(gty, gadget, {}, "Raw");

    Value* out = nullptr;
    Type* t = si.ValTy;
    if (t->isIntegerTy(64)) out = raw;
    else if (t->isIntegerTy()) out = b.CreateTrunc(raw, t);
    else if (t->isFloatTy())
    {
        Value* t32 = b.CreateTrunc(raw, i32);
        out = b.CreateBitCast(t32, t);
    }
    else if (t->isDoubleTy()) out = b.CreateBitCast(raw, t);
    else if (t->isPointerTy()) out = b.CreateIntToPtr(raw, t);

    if (!out) return;

    li->replaceAllUsesWith(out);
    li->eraseFromParent();
}

static void RewriteStore(OpSite& si, RuntimeRefs& r)
{
    auto* sti = cast<StoreInst>(si.Inst);
    LLVMContext& c = sti->getContext();
    IntegerType* i32 = Type::getInt32Ty(c);
    Type* i64 = Type::getInt64Ty(c);
    Type* voidt = Type::getVoidTy(c);

    IRBuilder<> b(sti);
    Value* slotarg = BuildAddrSlotArg(b, r, si.Target);
    Value* sidc = ConstantInt::get(i64, SafeEncSid(si.Sid, r.K1));
    Value* sizec = ConstantInt::get(i32, si.SizeBytes);

    Value* gadget = b.CreateCall(r.FnGetStoreGadget, {sidc, slotarg, sizec}, "Gadget");

    Value* val = sti->getValueOperand();
    Type* t = val->getType();
    Value* wide = nullptr;
    if (t->isIntegerTy(64)) wide = val;
    else if (t->isIntegerTy()) wide = b.CreateZExt(val, i64);
    else if (t->isFloatTy())
    {
        Value* bits = b.CreateBitCast(val, i32);
        wide = b.CreateZExt(bits, i64);
    }
    else if (t->isDoubleTy()) wide = b.CreateBitCast(val, i64);
    else if (t->isPointerTy()) wide = b.CreatePtrToInt(val, i64);

    if (!wide) return;

    FunctionType* gty = FunctionType::get(voidt, {i64}, false);
    b.CreateCall(gty, gadget, {wide});

    sti->eraseFromParent();
}

static void RewriteLea(OpSite& si, RuntimeRefs& r)
{
    Use* u = si.LeaUse;
    User* usr = u->getUser();
    auto* inst = dyn_cast<Instruction>(usr);
    if (!inst) return;

    LLVMContext& c = inst->getContext();
    Type* i64 = Type::getInt64Ty(c);
    PointerType* ptrty = PointerType::get(c, 0);

    IRBuilder<> b(inst);
    Value* slotarg = BuildAddrSlotArg(b, r, si.Target);
    Value* sidc = ConstantInt::get(i64, SafeEncSid(si.Sid, r.K1));

    Value* gadget = b.CreateCall(r.FnGetLeaGadget, {sidc, slotarg}, "Gadget");
    FunctionType* gty = FunctionType::get(i64, {}, false);
    Value* raw = b.CreateCall(gty, gadget, {}, "Raw");
    Value* ptr = b.CreateIntToPtr(raw, ptrty, "LeaAddr");

    u->set(ptr);
}

static void CollectCallSites(Function& f, RuntimeRefs& r, std::vector<CallSite>& work)
{
    if (FunctionIsSkippable(f, r)) return;
    if (!r.JitropFns.count(&f)) return;
    unsigned siteidx = 0;
    for (BasicBlock& bb : f)
    {
        for (Instruction& i : bb)
        {
            auto* ci = dyn_cast<CallInst>(&i);
            if (!ci) continue;
            if (!CallSiteEligible(ci, r)) continue;
            uint64_t sid = ComputeSiteId(f, siteidx++, 'C');
            work.push_back({ci, sid});
            Function* callee = ci->getCalledFunction();
            if (callee && r.CalleeIndex.find(callee) == r.CalleeIndex.end())
                r.CalleeIndex[callee] = (unsigned)r.CalleeIndex.size();
        }
    }
}

static void CollectOpSites(Function& f, RuntimeRefs& r, std::vector<OpSite>& work)
{
    if (f.isDeclaration()) return;
    StringRef n = f.getName();
    if (IsOurRuntime(n)) return;
    if (n == "JITROPInitTables") return;
    if (r.NoJitropFns.count(&f)) return;

    const DataLayout& dl = f.getParent()->getDataLayout();
    bool funcopted = r.JitropFns.count(&f) > 0;
    unsigned idxl = 0, idxs = 0, idxa = 0;

    for (BasicBlock& bb : f)
    {
        for (Instruction& i : bb)
        {
            if (auto* li = dyn_cast<LoadInst>(&i))
            {
                auto* gv = dyn_cast<GlobalVariable>(li->getPointerOperand()->stripPointerCasts());
                if (!gv) continue;
                if (!r.JitropGvs.count(gv) && !funcopted) continue;
                if (li->isAtomic()) continue;
                unsigned sz = 0;
                if (!IsSupportedScalarType(li->getType(), dl, sz)) continue;
                OpSite si{};
                si.Kind = OpSite::KLoad;
                si.Inst = li;
                si.LeaUse = nullptr;
                si.Target = gv;
                si.SizeBytes = sz;
                si.ValTy = li->getType();
                si.Sid = ComputeSiteId(f, idxl++, 'L');
                work.push_back(si);
                if (r.AddrIndex.find(gv) == r.AddrIndex.end())
                    r.AddrIndex[gv] = (unsigned)r.AddrIndex.size();
            }
            else if (auto* sti = dyn_cast<StoreInst>(&i))
            {
                auto* gv = dyn_cast<GlobalVariable>(sti->getPointerOperand()->stripPointerCasts());
                if (!gv) continue;
                if (!r.JitropGvs.count(gv) && !funcopted) continue;
                if (sti->isAtomic()) continue;
                unsigned sz = 0;
                Type* vt = sti->getValueOperand()->getType();
                if (!IsSupportedScalarType(vt, dl, sz)) continue;
                OpSite si{};
                si.Kind = OpSite::KStore;
                si.Inst = sti;
                si.LeaUse = nullptr;
                si.Target = gv;
                si.SizeBytes = sz;
                si.ValTy = vt;
                si.Sid = ComputeSiteId(f, idxs++, 'S');
                work.push_back(si);
                if (r.AddrIndex.find(gv) == r.AddrIndex.end())
                    r.AddrIndex[gv] = (unsigned)r.AddrIndex.size();
            }
        }
    }

    // lea rewrites only fire for GV-scoped opt in; function-scoped opt in covers accesses INSIDE the function, not address taken uses of a GV that merely appears in it
    for (GlobalVariable* gv : r.JitropGvs)
    {
        std::vector<Use*> leauses;
        for (Use& u : gv->uses())
        {
            auto* usr = dyn_cast<Instruction>(u.getUser());
            if (!usr) continue;
            if (usr->getFunction() != &f) continue;
            if (auto* li = dyn_cast<LoadInst>(usr))
                if (li->getPointerOperandIndex() == u.getOperandNo()) continue;
            if (auto* sti = dyn_cast<StoreInst>(usr))
                if (sti->getPointerOperandIndex() == u.getOperandNo()) continue;
            leauses.push_back(&u);
        }
        for (Use* u : leauses)
        {
            OpSite si{};
            si.Kind = OpSite::KLea;
            si.Inst = dyn_cast<Instruction>(u->getUser());
            si.LeaUse = u;
            si.Target = gv;
            si.SizeBytes = 0;
            si.ValTy = nullptr;
            si.Sid = ComputeSiteId(f, idxa++, 'A');
            work.push_back(si);
            if (r.AddrIndex.find(gv) == r.AddrIndex.end())
                r.AddrIndex[gv] = (unsigned)r.AddrIndex.size();
        }
    }
}

static void EmitCalleeTable(Module& m, RuntimeRefs& r)
{
    LLVMContext& c = m.getContext();
    PointerType* ptrty = PointerType::get(c, 0);
    unsigned n = (unsigned)r.CalleeIndex.size();
    ArrayType* tyarr = ArrayType::get(ptrty, n);
    r.GvCalleeTable = new GlobalVariable(m, tyarr, false,
        GlobalValue::InternalLinkage,
        ConstantAggregateZero::get(tyarr), "JITROPCalleeTable");
    r.GvCalleeTable->setAlignment(Align(8));
   
    r.GvCalleeTable->setExternallyInitialized(true);
}

static void EmitAddrTable(Module& m, RuntimeRefs& r)
{
    LLVMContext& c = m.getContext();
    PointerType* ptrty = PointerType::get(c, 0);
    unsigned n = (unsigned)r.AddrIndex.size();
    ArrayType* tyarr = ArrayType::get(ptrty, n);
    r.GvAddrTable = new GlobalVariable(m, tyarr, false,
        GlobalValue::InternalLinkage,
        ConstantAggregateZero::get(tyarr), "JITROPAddrTable");
    r.GvAddrTable->setAlignment(Align(8));
    r.GvAddrTable->setExternallyInitialized(true);
}

static void EmitSiteCache(Module& m, RuntimeRefs& r, unsigned n)
{
    LLVMContext& c = m.getContext();
    PointerType* ptrty = PointerType::get(c, 0);
    ArrayType* tyarr = ArrayType::get(ptrty, n);
    r.GvSiteCache = new GlobalVariable(m, tyarr, false,
        GlobalValue::InternalLinkage,
        ConstantAggregateZero::get(tyarr), "JITROPSiteCache");
    r.GvSiteCache->setAlignment(Align(8));
    r.GvSiteCache->setExternallyInitialized(true);
}

static void EmitTablesInit(Module& m, RuntimeRefs& r)
{
    LLVMContext& c = m.getContext();
    Type* voidt = Type::getVoidTy(c);
    PointerType* ptrty = PointerType::get(c, 0);
    Type* i64 = Type::getInt64Ty(c);

    FunctionType* ft = FunctionType::get(voidt, {}, false);
    r.FnInitTables = Function::Create(ft, GlobalValue::InternalLinkage, "JITROPInitTables", &m);
    r.FnInitTables->addFnAttr(Attribute::NoInline);

    BasicBlock* entry = BasicBlock::Create(c, "Entry", r.FnInitTables);
    IRBuilder<> b(entry);

    if (r.GvCalleeTable)
    {
        unsigned n = (unsigned)r.CalleeIndex.size();
        ArrayType* tyarr = ArrayType::get(ptrty, n);
        std::vector<std::pair<unsigned, Function*>> ordered;
        ordered.reserve(r.CalleeIndex.size());
        for (auto& kv : r.CalleeIndex) ordered.push_back({kv.second, kv.first});
        std::sort(ordered.begin(), ordered.end(),
            [](const auto& a, const auto& b) { return a.first < b.first; });
        for (auto& [idx, callee] : ordered)
        {
            Value* slot = b.CreateGEP(tyarr, r.GvCalleeTable,
                {ConstantInt::get(i64, 0), ConstantInt::get(i64, idx)});
            b.CreateAlignedStore(callee, slot, Align(8));
        }
    }

    if (r.GvAddrTable)
    {
        unsigned n = (unsigned)r.AddrIndex.size();
        ArrayType* tyarr = ArrayType::get(ptrty, n);
        std::vector<std::pair<unsigned, GlobalVariable*>> ordered;
        ordered.reserve(r.AddrIndex.size());
        for (auto& kv : r.AddrIndex) ordered.push_back({kv.second, kv.first});
        std::sort(ordered.begin(), ordered.end(),
            [](const auto& a, const auto& b) { return a.first < b.first; });
        for (auto& [idx, gv] : ordered)
        {
            Value* slot = b.CreateGEP(tyarr, r.GvAddrTable,
                {ConstantInt::get(i64, 0), ConstantInt::get(i64, idx)});
            b.CreateAlignedStore(gv, slot, Align(8));
        }
    }

    b.CreateRetVoid();

   
    appendToGlobalCtors(m, r.FnInitTables, InitCtorPriority);
}

bool RunJITROP(Module& m)
{
    if (!JITROPEnable) return false;

    RuntimeRefs r = EmitRuntimeSkeleton(m);
    CollectAnnotations(m, r);
    if (r.JitropFns.empty() && r.JitropGvs.empty()) return false;

    EmitGadgetScanBody(m, r.FnFindBareRet, r.GvBareRet, 0x08, 4, r);
    EmitGadgetScanBody(m, r.FnFindTramp40, r.GvTramp40, 0x28, 0, r);
    EmitGadgetScanBody(m, r.FnFindTramp56, r.GvTramp56, 0x38, 0, r);
    EmitGadgetScanBody(m, r.FnFindTramp72, r.GvTramp72, 0x48, 0, r);
    EmitGadgetScanBody(m, r.FnFindTramp88, r.GvTramp88, 0x58, 0, r);

    EmitBuildChainBasicBody(m, r);
    EmitBuildChainShadowBody(m, r);
    EmitBuildChainCopyBody(m, r);

    EmitGetChainBody(m.getContext(), r.FnGetChainBasic, r.FnBuildChainBasic,
        r.GvCacheBasic, r.CacheEntryTy, r.CacheArrTy, r.K2, false);
    EmitGetChainBody(m.getContext(), r.FnGetChainShadow, r.FnBuildChainShadow,
        r.GvCacheShadow, r.CacheEntryTy, r.CacheArrTy, r.K2, false);
    EmitGetChainBody(m.getContext(), r.FnGetChainCopy, r.FnBuildChainCopy,
        r.GvCacheCopy, r.CacheEntryTy, r.CacheArrTy, r.K2, true);

    EmitBuildLoadGadgetBody(m, r);
    EmitBuildStoreGadgetBody(m, r);
    EmitBuildLeaGadgetBody(m, r);

    EmitGetGadgetBody(m.getContext(), r.FnGetLoadGadget, r.FnBuildLoadGadget,
        r.GvCacheLoad, r.CacheEntryTy, r.CacheArrTy, r.K2, true);
    EmitGetGadgetBody(m.getContext(), r.FnGetStoreGadget, r.FnBuildStoreGadget,
        r.GvCacheStore, r.CacheEntryTy, r.CacheArrTy, r.K2, true);
    EmitGetGadgetBody(m.getContext(), r.FnGetLeaGadget, r.FnBuildLeaGadget,
        r.GvCacheLea, r.CacheEntryTy, r.CacheArrTy, r.K2, false);

    std::vector<Function*> funcs;
    funcs.reserve(m.size());
    for (Function& f : m) funcs.push_back(&f);

    std::vector<CallSite> callwork;
    std::vector<OpSite> opwork;
    
    for (Function* f : funcs)
    {
        CollectCallSites(*f, r, callwork);
        CollectOpSites(*f, r, opwork);
    }

    if (!r.CalleeIndex.empty()) EmitCalleeTable(m, r);
    if (!r.AddrIndex.empty()) EmitAddrTable(m, r);
    if (r.GvCalleeTable || r.GvAddrTable) EmitTablesInit(m, r);

    if (!callwork.empty()) EmitSiteCache(m, r, (unsigned)callwork.size());

    for (size_t i = 0; i < callwork.size(); ++i)
        RewriteCallSite(callwork[i].Call, callwork[i].Sid, (unsigned)i, r);

   
    for (auto& si : opwork)
    {
        if (si.Kind == OpSite::KLoad) RewriteLoad(si, r);
        else if (si.Kind == OpSite::KStore) RewriteStore(si, r);
    }
    for (auto& si : opwork)
    {
        if (si.Kind == OpSite::KLea) RewriteLea(si, r);
    }

    return !callwork.empty() || !opwork.empty();
}

PreservedAnalyses JITROPPass::run(Module& m, ModuleAnalysisManager&)
{
    bool mutated = RunJITROP(m);
    return mutated ? PreservedAnalyses::none() : PreservedAnalyses::all();
}