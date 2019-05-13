//
// Created by kier on 19-5-13.
//

#include "api/intime.h"

#include "declare_tensor.h"

#include "frontend/intime.h"

using namespace ts;

ts_Tensor *ts_intime_transpose(const ts_Tensor *x, const int32_t *permute, int32_t len) {
    TRY_HEAD
        if (!x) throw Exception("NullPointerException: @param: 1");
        if (!permute) throw Exception("NullPointerException: @param: 2");
        std::unique_ptr<ts_Tensor> dolly(new ts_Tensor(
                intime::transpose(**x, std::vector<int32_t>(permute, permute + len))
        ));
    RETURN_OR_CATCH(dolly.release(), nullptr)
}

ts_Tensor *ts_intime_sigmoid(const ts_Tensor *x) {
    TRY_HEAD
        if (!x) throw Exception("NullPointerException: @param: 1");
        std::unique_ptr<ts_Tensor> dolly(new ts_Tensor(
                intime::sigmoid(**x)
        ));
    RETURN_OR_CATCH(dolly.release(), nullptr)
}
