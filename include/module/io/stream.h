//
// Created by kier on 2018/11/7.
//

#ifndef TENSORSTACK_MODULE_IO_STREAM_H
#define TENSORSTACK_MODULE_IO_STREAM_H


#include <cstddef>

namespace ts {
    class StreamReader {
    public:
        using self = StreamReader;

        virtual size_t read(void *buffer, size_t size) = 0;
    };

    class StreamWriter {
    public:
        using self = StreamWriter;

        virtual size_t write(const void *buffer, size_t size) = 0;
    };

    class Stream : public StreamWriter, public StreamReader {
    public:
        using self = Stream;
    };
}


#endif //TENSORSTACK_MODULE_IO_STREAM_H
