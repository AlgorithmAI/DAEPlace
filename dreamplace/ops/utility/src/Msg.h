#ifndef DREAMPLACE_MSG_H
#define DREAMPLACE_MSG_H

#include <cstdarg>
#include <cassert>
#include <cstdio>
#include <cstdlib>
#include "utility/src/Namespace.h"
//#define_CRT_SECURE_NO_WARNINGS
DREAMPLACE_BEGIN_NAMESPACE

enum MessageType {
    kNONE = 0,
    kINFO = 1,
    kWARN = 2,
    kERROR = 3,
    kDEBUG = 4,
    kASSERT = 5
};
///stdout，输出到屏幕
int dreamplacePrint(MessageType m, const char* format, ...);
///打印到流上
int dreamplacePrintStream(MessageType m, FILE* stream, const char* format, ...);
///要实现一个核心函数，用于从可变参数列表中打印格式化数据
int dreamplaceVPrintStream(MessageType m, FILE* stream, const char* format, va_list args);
///格式输出到缓冲区
int dreamplaceSPrint(MessageType m, char* buf, const char* format, ...);
///核心函数，实现格式化输出到缓冲区
int dreamplaceVSPrint(MessageType m, char* buf, const char* format, va_list args);
///格式前缀
int dreamplaceSPrintPrefix(MessageType m, char* buf);

///断言
void dreamplacePrintAssertMsg(const char* expr, const char* fileName, unsigned lineNum, const char* funcName, const char* format, ...);
void dreamplacePrintAssertMsg(const char* expr, const char* fileName, unsigned lineNum, const char* funcName);

#define dreamplaceAssertMsg(condition, args...) do {\
    if(!(condition)) \
    {\
        ::DREAMPLACE_NAMESPACE::dreamplacePrintAssertMsg(#condition, __FILE__, __LINE__, __PRETTY_FUNCTION__, args); \
        abort(); \
    }\
}while(false)
#define dreamplaceAssert(condition) do {\
     if(!(condition)) \
     {\
       ::DREAMPLACE_NAMESPACE::dreamplacePrintAssertMsg(#condition, __FILE__, __LINE__, __PRETTY_FUNCTION__); \
       abort(); \
     }\
}while(false)

///静态断言
template <bool>
struct dreamplaceStaticAssert;
template <>
struct dreamplaceStaticAssert<true>
{
    dreamplaceStaticAssert(const char* = NULL) {}
};


DREAMPLACE_END_NAMESPACE

#endif