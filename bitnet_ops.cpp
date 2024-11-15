// bitnet_ops.cpp
#include <torch/extension.h>
#include <vector>
#include <thread>
#include <cmath>

// Create lookup table at compile time
constexpr int8_t get_unpacked_value(uint8_t val) {
    return (val == 0b10) ? 1 : ((val == 0b11) ? -1 : 0);
}

constexpr std::array<std::array<int8_t, 4>, 256> generate_lut() {
    std::array<std::array<int8_t, 4>, 256> lut = {};
    for (int i = 0; i < 256; i++) {
        for (int j = 0; j < 4; j++) {
            lut[i][j] = get_unpacked_value((i >> (j * 2)) & 0b11);
        }
    }
    return lut;
}

static const auto UNPACK_LUT = generate_lut();

torch::Tensor pack_weights(const torch::Tensor& weights) {
    TORCH_CHECK(weights.dim() == 2, "Weights must be a 2D tensor");
    
    auto in_features = weights.size(1);
    auto out_features = weights.size(0);
    auto weights_cpu = weights.to(torch::kCPU);
    
    int64_t num_elements = in_features * out_features;
    int64_t packed_size = (num_elements + 3) / 4;
    
    auto packed = torch::zeros({packed_size}, torch::kUInt8);
    auto packed_ptr = packed.data_ptr<uint8_t>();
    auto weights_ptr = weights_cpu.data_ptr<float>();
    
    for (int64_t i = 0; i < num_elements; i += 4) {
        uint8_t packed_byte = 0;
        for (int j = 0; j < 4 && (i + j) < num_elements; j++) {
            float val = weights_ptr[i + j];
            uint8_t packed_val;
            if (val > 0.5f) packed_val = 0b10;      // 1
            else if (val < -0.5f) packed_val = 0b11; // -1
            else packed_val = 0b00;                  // 0
            packed_byte |= (packed_val << (j * 2));
        }
        packed_ptr[i / 4] = packed_byte;
    }
    
    return packed;
}

torch::Tensor unpack_weights(const torch::Tensor& packed, int64_t in_features, int64_t out_features) {
    auto options = torch::TensorOptions().dtype(torch::kInt8).device(packed.device());
    auto unpacked = torch::zeros({out_features, in_features}, options);
    auto unpacked_ptr = unpacked.data_ptr<int8_t>();
    auto packed_ptr = packed.data_ptr<uint8_t>();
    
    int64_t num_elements = in_features * out_features;
    int64_t num_bytes = (num_elements + 3) / 4;
    
    const int num_threads = std::thread::hardware_concurrency();
    std::vector<std::thread> threads;
    
    auto unpack_chunk = [&](int64_t start, int64_t end) {
        for (int64_t i = start; i < end; i++) {
            uint8_t packed_byte = packed_ptr[i];
            const auto& unpacked_values = UNPACK_LUT[packed_byte];
            
            int64_t out_idx = i * 4;
            if (out_idx + 3 < num_elements) {
                unpacked_ptr[out_idx] = unpacked_values[0];
                unpacked_ptr[out_idx + 1] = unpacked_values[1];
                unpacked_ptr[out_idx + 2] = unpacked_values[2];
                unpacked_ptr[out_idx + 3] = unpacked_values[3];
            } else {
                for (int j = 0; j < 4 && out_idx + j < num_elements; j++) {
                    unpacked_ptr[out_idx + j] = unpacked_values[j];
                }
            }
        }
    };
    
    int64_t chunk_size = (num_bytes + num_threads - 1) / num_threads;
    for (int i = 0; i < num_threads; i++) {
        int64_t start = i * chunk_size;
        int64_t end = std::min(start + chunk_size, num_bytes);
        if (start < end) {
            threads.emplace_back(unpack_chunk, start, end);
        }
    }
    
    for (auto& thread : threads) {
        thread.join();
    }
    
    return unpacked;
}

torch::Tensor activation_quant(const torch::Tensor& x, int64_t num_bits) {
    auto dtype = x.scalar_type();
    auto x_float = x.to(torch::kFloat32);
    
    auto Qn = -std::pow(2, num_bits - 1);
    auto Qp = std::pow(2, num_bits - 1) - 1;
    
    auto abs_x = x_float.abs();
    auto max_vals = std::get<0>(abs_x.max(-1, true));
    auto s = Qp / max_vals.clamp(1e-5);
    
    auto result = ((x_float * s).round().clamp(Qn, Qp)) / s;
    return result.to(dtype);
}

torch::Tensor bitnet_forward(const torch::Tensor& input, 
                           const torch::Tensor& packed_weights,
                           const c10::optional<torch::Tensor>& bias,
                           int64_t in_features,
                           int64_t out_features,
                           int64_t input_bits,
                           float scale) {
    auto quant_input = input + (activation_quant(input, input_bits) - input).detach();
    auto weights = unpack_weights(packed_weights, in_features, out_features);
    auto output = torch::mm(quant_input, weights.to(torch::kFloat32).t());
    
    if (bias.has_value()) {
        output += bias.value().view({1, -1});
    }
    
    return output / scale;
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("pack_weights", &pack_weights, "Pack weights into 2-bit representation");
    m.def("unpack_weights", &unpack_weights, "Unpack weights from 2-bit representation");
    m.def("bitnet_forward", &bitnet_forward, "BitNet forward pass");
}