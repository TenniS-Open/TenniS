#include "rknn_dll.h"

#include <stdexcept>

#include "seeta_aip/seeta_aip_dll.h"

RknnDll::RknnDll() {
#ifdef RKNN_BUILTIN
    this->init = rknn_init;
    this->dup_context = rknn_dup_context;
    this->destroy = rknn_destroy;
    this->query = rknn_query;
    this->inputs_set = rknn_inputs_set;
    this->set_batch_core_num = rknn_set_batch_core_num;
    this->set_core_mask = rknn_set_core_mask;
    this->run = rknn_run;
    this->wait = rknn_wait;
    this->outputs_get = rknn_outputs_get;
    this->outputs_release = rknn_outputs_release;
    this->create_mem_from_phys = rknn_create_mem_from_phys;
    this->create_mem_from_fd = rknn_create_mem_from_fd;
    // this->create_mem_from_mb_blk = rknn_create_mem_from_mb_blk;
    this->create_mem = rknn_create_mem;
    this->destroy_mem = rknn_destroy_mem;
    this->set_weight_mem = rknn_set_weight_mem;
    this->set_internal_mem = rknn_set_internal_mem;
    this->set_io_mem = rknn_set_io_mem;
#else
    m_dll = seeta::aip::dlopen_v2("rknnrt");
    if (!m_dll) m_dll = seeta::aip::dlopen_v2("rknn_api");
    if (!m_dll) throw std::logic_error(std::string("Can not open dll rknnrt or rknn_api. ") + seeta::aip::dlerror());

    this->init = (decltype(rknn_init) *) seeta::aip::dlsym(m_dll, "rknn_init");
    this->dup_context = (decltype(rknn_dup_context) *) seeta::aip::dlsym(m_dll, "rknn_dup_context");
    this->destroy = (decltype(rknn_destroy) *) seeta::aip::dlsym(m_dll, "rknn_destroy");
    this->query = (decltype(rknn_query) *) seeta::aip::dlsym(m_dll, "rknn_query");
    this->inputs_set = (decltype(rknn_inputs_set) *) seeta::aip::dlsym(m_dll, "rknn_inputs_set");
    this->set_batch_core_num = (decltype(rknn_set_batch_core_num) *) seeta::aip::dlsym(m_dll,
                                                                                       "rknn_set_batch_core_num");
    this->set_core_mask = (decltype(rknn_set_core_mask) *) seeta::aip::dlsym(m_dll, "rknn_set_core_mask");
    this->run = (decltype(rknn_run) *) seeta::aip::dlsym(m_dll, "rknn_run");
    this->wait = (decltype(rknn_wait) *) seeta::aip::dlsym(m_dll, "rknn_wait");
    this->outputs_get = (decltype(rknn_outputs_get) *) seeta::aip::dlsym(m_dll, "rknn_outputs_get");
    this->outputs_release = (decltype(rknn_outputs_release) *) seeta::aip::dlsym(m_dll, "rknn_outputs_release");
    this->create_mem_from_phys = (decltype(rknn_create_mem_from_phys) *) seeta::aip::dlsym(m_dll,
                                                                                           "rknn_create_mem_from_phys");
    this->create_mem_from_fd = (decltype(rknn_create_mem_from_fd) *) seeta::aip::dlsym(m_dll,
                                                                                       "rknn_create_mem_from_fd");
    this->create_mem_from_mb_blk = (decltype(rknn_create_mem_from_mb_blk) *) seeta::aip::dlsym(m_dll,
                                                                                               "rknn_create_mem_from_mb_blk");
    this->create_mem = (decltype(rknn_create_mem) *) seeta::aip::dlsym(m_dll, "rknn_create_mem");
    this->destroy_mem = (decltype(rknn_destroy_mem) *) seeta::aip::dlsym(m_dll, "rknn_destroy_mem");
    this->set_weight_mem = (decltype(rknn_set_weight_mem) *) seeta::aip::dlsym(m_dll, "rknn_set_weight_mem");
    this->set_internal_mem = (decltype(rknn_set_internal_mem) *) seeta::aip::dlsym(m_dll, "rknn_set_internal_mem");
    this->set_io_mem = (decltype(rknn_set_io_mem) *) seeta::aip::dlsym(m_dll, "rknn_set_io_mem");
#endif
}

RknnDll::~RknnDll() {
    if (m_dll) {
        seeta::aip::dlclose(m_dll);
    }
}
