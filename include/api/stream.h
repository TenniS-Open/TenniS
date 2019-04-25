//
// Created by kier on 19-4-25.
//

#ifndef TENSORSTACK_API_STREAM_H
#define TENSORSTACK_API_STREAM_H

#ifdef __cplusplus
extern "C" {
#endif

#include <stdint.h>
#include <stddef.h>

typedef uint64_t ts_stream_write(void *obj, const char *data, uint64_t length);
typedef uint64_t ts_stream_read(void *obj, char *data, uint64_t length);

#ifdef __cplusplus
}
#endif

#endif //TENSORSTACK_API_STREAM_H
