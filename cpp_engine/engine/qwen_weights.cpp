// CUDA-side of the Qwen weight loader: device residency and host-to-device
// upload. The checkpoint mapping, TP sharding and host materialization live in
// core/qwen_weight_map.cpp and stay free of any vendor SDK, so a checkpoint can
// be audited on a machine that has no CUDA toolkit at all.

#include "qwen_weights.hpp"

#include <cuda_runtime.h>

#include <stdexcept>

namespace dsv4 {

QwenDeviceTensor::~QwenDeviceTensor() {
    if (data != nullptr) cudaFree(data);
}

QwenDeviceTensor::QwenDeviceTensor(QwenDeviceTensor&& other) noexcept
    : data(other.data), device_dtype(other.device_dtype), shape(std::move(other.shape)),
      nbytes(other.nbytes), capacity(other.capacity) {
    other.data = nullptr;
    other.nbytes = 0;
    other.capacity = 0;
    other.device_dtype = SafeDType::Unknown;
}

QwenDeviceTensor& QwenDeviceTensor::operator=(QwenDeviceTensor&& other) noexcept {
    if (this == &other) return *this;
    if (data != nullptr) cudaFree(data);
    data = other.data;
    device_dtype = other.device_dtype;
    shape = std::move(other.shape);
    nbytes = other.nbytes;
    capacity = other.capacity;
    other.data = nullptr;
    other.nbytes = 0;
    other.capacity = 0;
    other.device_dtype = SafeDType::Unknown;
    return *this;
}

float* QwenDeviceTensor::f32_data() {
    if (device_dtype != SafeDType::F32) throw std::runtime_error("Qwen tensor is not F32");
    return static_cast<float*>(data);
}

const float* QwenDeviceTensor::f32_data() const {
    if (device_dtype != SafeDType::F32) throw std::runtime_error("Qwen tensor is not F32");
    return static_cast<const float*>(data);
}

uint16_t* QwenDeviceTensor::f16_data() {
    if (device_dtype != SafeDType::F16) throw std::runtime_error("Qwen tensor is not F16");
    return static_cast<uint16_t*>(data);
}

const uint16_t* QwenDeviceTensor::f16_data() const {
    if (device_dtype != SafeDType::F16) throw std::runtime_error("Qwen tensor is not F16");
    return static_cast<const uint16_t*>(data);
}

uint8_t* QwenDeviceTensor::fp8_data() {
    if (device_dtype != SafeDType::F8_E4M3) throw std::runtime_error("Qwen tensor is not FP8 E4M3");
    return static_cast<uint8_t*>(data);
}

const uint8_t* QwenDeviceTensor::fp8_data() const {
    if (device_dtype != SafeDType::F8_E4M3) throw std::runtime_error("Qwen tensor is not FP8 E4M3");
    return static_cast<const uint8_t*>(data);
}

uint8_t* QwenDeviceTensor::byte_data() {
    if (device_dtype != SafeDType::I8) throw std::runtime_error("Qwen tensor is not raw bytes");
    return static_cast<uint8_t*>(data);
}

const uint8_t* QwenDeviceTensor::byte_data() const {
    if (device_dtype != SafeDType::I8) throw std::runtime_error("Qwen tensor is not raw bytes");
    return static_cast<const uint8_t*>(data);
}

int8_t* QwenDeviceTensor::int8_data() {
    if (device_dtype != SafeDType::I8) throw std::runtime_error("Qwen tensor is not INT8");
    return static_cast<int8_t*>(data);
}

const int8_t* QwenDeviceTensor::int8_data() const {
    if (device_dtype != SafeDType::I8) throw std::runtime_error("Qwen tensor is not INT8");
    return static_cast<const int8_t*>(data);
}

uint8_t* QwenDeviceTensor::u8_data() {
    if (device_dtype != SafeDType::U8) throw std::runtime_error("Qwen tensor is not U8");
    return static_cast<uint8_t*>(data);
}

const uint8_t* QwenDeviceTensor::u8_data() const {
    if (device_dtype != SafeDType::U8) throw std::runtime_error("Qwen tensor is not U8");
    return static_cast<const uint8_t*>(data);
}

QwenDeviceTensor qwen_upload_tensor_cuda(const SafeTensorsIndex& index,
                                        const QwenTensorRef& ref,
                                        void* stream) {
    QwenHostTensor host = qwen_materialize_host_tensor(index, ref);
    QwenDeviceTensor device;
    device.device_dtype = host.device_dtype;
    device.shape = host.shape;
    device.nbytes = host.bytes.size();
    device.capacity = device.nbytes;
    if (device.nbytes == 0 || cudaMalloc(&device.data, device.nbytes) != cudaSuccess) {
        throw std::runtime_error("failed to allocate Qwen device tensor: " + ref.name);
    }
    const cudaStream_t cuda_stream = static_cast<cudaStream_t>(stream);
    const cudaError_t status = cudaMemcpyAsync(device.data, host.bytes.data(), device.nbytes,
                                               cudaMemcpyHostToDevice, cuda_stream);
    if (status != cudaSuccess) {
        cudaFree(device.data);
        device.data = nullptr;
        device.nbytes = 0;
        device.capacity = 0;
        throw std::runtime_error("failed to upload Qwen device tensor: " + ref.name);
    }
    return device;
}

QwenDeviceTensor qwen_upload_nvfp4_linear_cuda(
    const SafeTensorsIndex& index, const QwenLinearRef& ref,
    float* weight_global_factor, float* input_global_scale, void* stream) {
    if (weight_global_factor == nullptr || input_global_scale == nullptr) {
        throw std::invalid_argument("null Qwen NVFP4 global metadata output");
    }
    QwenNvfp4HostLinear host = qwen_materialize_nvfp4_host_linear(index, ref);
    QwenDeviceTensor device;
    device.device_dtype = SafeDType::U8;
    device.shape = host.logical_shape;
    device.shape.push_back(sizeof(QwenNvfp4Block64));
    device.nbytes = host.blocks.size() * sizeof(QwenNvfp4Block64);
    device.capacity = device.nbytes;
    if (device.nbytes == 0 || cudaMalloc(&device.data, device.nbytes) != cudaSuccess) {
        throw std::runtime_error("failed to allocate Qwen NVFP4 device linear");
    }
    const cudaStream_t cuda_stream = static_cast<cudaStream_t>(stream);
    const cudaError_t status = cudaMemcpyAsync(
        device.data, host.blocks.data(), device.nbytes, cudaMemcpyHostToDevice,
        cuda_stream);
    if (status != cudaSuccess) {
        cudaFree(device.data);
        device.data = nullptr;
        device.nbytes = 0;
        device.capacity = 0;
        throw std::runtime_error("failed to upload Qwen NVFP4 device linear");
    }
    *weight_global_factor = host.weight_global_factor;
    *input_global_scale = host.input_global_scale;
    return device;
}

}  // namespace dsv4
