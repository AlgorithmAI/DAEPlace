/*************************************************************************
    > File Name: Msg.cpp
    > Author: Xu Li
    > Mail: lixu@cnic.cn
    > Created Time: Fri 31 Jul 2024 06:20:14 PM CDT
 ************************************************************************/


#include <cstdlib>
#include <cstdio>
#include <cstring>
#include "utility/src/Msg.h"

DREAMPLACE_BEGIN_NAMESPACE

int dreamplacePrint(MessageType m, const char* format, ...)
{
    va_list args;
    va_start(args, format);
    int ret = dreamplaceVPrintStream(m, stdout, format, args);
    va_end(args);

    return ret;
}

int dreamplacePrintStream(MessageType m, FILE* stream, const char* format, ...)
{
    va_list args;
    va_start(args, format);
    int ret = dreamplaceVPrintStream(m, stream, format, args);
    va_end(args);

    return ret;
}

int dreamplaceVPrintStream(MessageType m, FILE* stream, const char* format, va_list args)
{
    //打印前缀
    char prefix[8];
    dreamplaceSPrintPrefix(m, prefix);
    fprintf(stream, "%s", prefix);

    //打印消息
    int ret = vfprintf(stream, format, args);

    return ret;
}

int dreamplaceSPrint(MessageType m, char* buf, const char* format, ...)
{
    va_list args;
    va_start(args, format);
    int ret = dreamplaceVSPrint(m, buf, format, args);
    va_end(args);

    return ret;
}

int dreamplaceVSPrint(MessageType m, char* buf, const char* format, va_list args)
{
    //打印前缀
    char prefix[8];
    dreamplaceSPrintPrefix(m, prefix);
    sprintf(buf, "%s", prefix);

    //打印消息
    int ret = vsprintf(buf + strlen(prefix), format, args);

    return ret;
}

int dreamplaceSPrintPrefix(MessageType m, char* prefix)
{
    switch (m)
    {
        case kNONE:
            return sprintf(prefix, "%c", '\0');
        case kINFO:
            return sprintf(prefix, "(I) ");
        case kWARN:
            return sprintf(prefix, "(W) ");
        case kERROR:
            return sprintf(prefix, "(E) ");
        case kDEBUG:
            return sprintf(prefix, "(D) ");
        case kASSERT:
            return sprintf(prefix, "(A) ");
        default:
            dreamplaceAssertMsg(0, "unknown message type");

    }
    return 0;
}

void dreamplacePrintAssertMsg(const char* expr, const char* fileName, unsigned lineNum, const char* funcName, const char* format, ...)
{
    // construct message
    char buf[1024];
    va_list args;
    va_start(args, format);
    vsprintf(buf, format, args);
    va_end(args);

    // print message
    dreamplacePrintStream(kASSERT, stderr, "%s:%u: %s: Assertion `%s' failed: %s\n", fileName, lineNum, funcName, expr, buf);
}

void dreamplacePrintAssertMsg(const char* expr, const char* fileName, unsigned lineNum, const char* funcName)
{
    //print message
    dreamplacePrintStream(kASSERT, stderr, "%s:%u: %s: Assertion '%s' failed\n", fileName, lineNum, funcName, expr);
}

DREAMPLACE_END_NAMESPACE

