#include "v2/rknn_api.h"

class RknnDll {
public:
    RknnDll();
    ~RknnDll();

    decltype(rknn_init) *init = nullptr;
    decltype(rknn_dup_context) *dup_context = nullptr;
    decltype(rknn_destroy) *destroy = nullptr;
    decltype(rknn_query) *query = nullptr;
    decltype(rknn_inputs_set) *inputs_set = nullptr;
    decltype(rknn_set_batch_core_num) *set_batch_core_num = nullptr;
    decltype(rknn_set_core_mask) *set_core_mask = nullptr;
    decltype(rknn_run) *run = nullptr;
    decltype(rknn_wait) *wait = nullptr;
    decltype(rknn_outputs_get) *outputs_get = nullptr;
    decltype(rknn_outputs_release) *outputs_release = nullptr;
    decltype(rknn_create_mem_from_phys) *create_mem_from_phys = nullptr;
    decltype(rknn_create_mem_from_fd) *create_mem_from_fd = nullptr;
    decltype(rknn_create_mem_from_mb_blk) *create_mem_from_mb_blk = nullptr;
    decltype(rknn_create_mem) *create_mem = nullptr;
    decltype(rknn_destroy_mem) *destroy_mem = nullptr;
    decltype(rknn_set_weight_mem) *set_weight_mem = nullptr;
    decltype(rknn_set_internal_mem) *set_internal_mem = nullptr;
    decltype(rknn_set_io_mem) *set_io_mem = nullptr;
private:
    void *m_dll = nullptr;
};
