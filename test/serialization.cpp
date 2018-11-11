//
// Created by kier on 2018/11/11.
//

#include <module/io/fstream.h>
#include <core/tensor.h>
#include <core/tensor_builder.h>
#include <utils/log.h>
#include <global/setup.h>

int main() {
    ts::setup();

    ts::Tensor str = ts::tensor::from("ABC");

    ts::FileStreamWriter out("test.bin");
    str.serialize(out);
    out.close();

    ts::Tensor temp;
    ts::FileStreamReader in("test.bin");
    temp.externalize(in);

    TS_LOG(ts::LOG_INFO) << ts::tensor::to_string(temp);
}

