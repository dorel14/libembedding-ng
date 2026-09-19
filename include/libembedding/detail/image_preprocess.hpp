/*
 * libembedding - detail/image_preprocess.hpp
 * Image preprocessing pipeline (resize, crop, normalize, HWC->CHW)
 *
 * Auteur: David Orel
 * Version: 1.6.0
 *
 * SPDX-License-Identifier: MIT
 */

#ifndef LIBEMBEDDING_DETAIL_IMAGE_PREPROCESS_HPP
#define LIBEMBEDDING_DETAIL_IMAGE_PREPROCESS_HPP

#ifndef LIBEMBEDDING_NO_IMAGE

#include <cmath>
#include <cstdint>
#include <cstring>
#include <stdexcept>
#include <string>
#include <vector>

/* stb_image and stb_image_resize are expected to be available */
#include "stb/stb_image.h"
#include "stb/stb_image_resize2.h"

namespace lembed { namespace detail {

/* Image data in CHW float format */
struct ImageTensor {
    std::vector<float> data; /* CHW layout */
    int channels;
    int height;
    int width;
};

/* Default ImageNet normalization constants */
static const float IMAGE_NET_MEAN[3] = { 0.485f, 0.456f, 0.406f };
static const float IMAGE_NET_STD[3]  = { 0.229f, 0.224f, 0.225f };

/* Convert RGB byte buffer to normalized CHW float format.
 * Output is CHW (planar) which requires stride jumps between channels.
 * This is the standard format expected by most vision models (ONNX).
 * The loop processes one pixel at a time (R,G,B) which is natural for RGB input. */
inline void rgb_to_chw_float(
        const unsigned char* rgb_data,
        int src_w,
        int crop_x, int crop_y,
        int target_size,
        float* out,
        const float mean[3],
        const float std_dev[3])
{
    const int stride = target_size * target_size;
    for (int y = 0; y < target_size; y++) {
        const unsigned char* src_row = rgb_data + ((crop_y + y) * src_w + crop_x) * 3;
        for (int x = 0; x < target_size; x++) {
            const unsigned char* src_px = src_row + x * 3;
            out[0 * stride + y * target_size + x] = ((float)src_px[0] / 255.0f - mean[0]) / std_dev[0];
            out[1 * stride + y * target_size + x] = ((float)src_px[1] / 255.0f - mean[1]) / std_dev[1];
            out[2 * stride + y * target_size + x] = ((float)src_px[2] / 255.0f - mean[2]) / std_dev[2];
        }
    }
}

/* Load image from file, resize to target_size, center crop, normalize, convert to CHW */
inline ImageTensor load_and_preprocess_image(
        const std::string& path,
        int target_size = 224,
        const float mean[3] = nullptr,
        const float std_dev[3] = nullptr) {

    const float* effective_mean = mean ? mean : IMAGE_NET_MEAN;
    const float* effective_std = std_dev ? std_dev : IMAGE_NET_STD;

    int w, h, c;
    unsigned char* img = stbi_load(path.c_str(), &w, &h, &c, 3);
    if (!img) throw std::runtime_error("Failed to load image: " + path);
    c = 3;

    /* Resize shorter side to target_size, maintaining aspect ratio */
    int new_w, new_h;
    if (w < h) { new_w = target_size; new_h = (int)((float)h / w * target_size); }
    else { new_h = target_size; new_w = (int)((float)w / h * target_size); }

    std::vector<unsigned char> resized(new_w * new_h * 3);
    stbir_resize_uint8_linear(img, w, h, 0, resized.data(), new_w, new_h, 0, (stbir_pixel_layout)3);
    stbi_image_free(img);

    int crop_x = (new_w - target_size) / 2;
    int crop_y = (new_h - target_size) / 2;

    ImageTensor result;
    result.channels = 3;
    result.height = target_size;
    result.width = target_size;
    result.data.resize(3 * target_size * target_size);

    rgb_to_chw_float(resized.data(), new_w, crop_x, crop_y, target_size,
                     result.data.data(), effective_mean, effective_std);

    return result;
}

/* Load image from memory buffer */
inline ImageTensor load_and_preprocess_image_bytes(
        const unsigned char* data, int data_size,
        int target_size = 224,
        const float mean[3] = nullptr,
        const float std_dev[3] = nullptr) {

    const float* effective_mean = mean ? mean : IMAGE_NET_MEAN;
    const float* effective_std = std_dev ? std_dev : IMAGE_NET_STD;

    int w, h, c;
    unsigned char* img = stbi_load_from_memory(data, data_size, &w, &h, &c, 3);
    if (!img) throw std::runtime_error("Failed to decode image from memory");
    c = 3;

    int new_w, new_h;
    if (w < h) { new_w = target_size; new_h = (int)((float)h / w * target_size); }
    else { new_h = target_size; new_w = (int)((float)w / h * target_size); }

    std::vector<unsigned char> resized(new_w * new_h * 3);
    stbir_resize_uint8_linear(img, w, h, 0, resized.data(), new_w, new_h, 0, (stbir_pixel_layout)3);
    stbi_image_free(img);

    int crop_x = (new_w - target_size) / 2;
    int crop_y = (new_h - target_size) / 2;

    ImageTensor result;
    result.channels = 3;
    result.height = target_size;
    result.width = target_size;
    result.data.resize(3 * target_size * target_size);

    rgb_to_chw_float(resized.data(), new_w, crop_x, crop_y, target_size,
                     result.data.data(), effective_mean, effective_std);

    return result;
}

/* Write preprocessed image from file directly into output buffer (zero-copy) */
inline void preprocess_image_to_buffer(
        const std::string& path,
        float* out,
        int target_size = 224,
        const float mean[3] = nullptr,
        const float std_dev[3] = nullptr) {

    const float* effective_mean = mean ? mean : IMAGE_NET_MEAN;
    const float* effective_std = std_dev ? std_dev : IMAGE_NET_STD;

    int w, h, c;
    unsigned char* img = stbi_load(path.c_str(), &w, &h, &c, 3);
    if (!img) throw std::runtime_error("Failed to load image: " + path);
    c = 3;

    int new_w, new_h;
    if (w < h) { new_w = target_size; new_h = (int)((float)h / w * target_size); }
    else { new_h = target_size; new_w = (int)((float)w / h * target_size); }

    std::vector<unsigned char> resized(new_w * new_h * 3);
    stbir_resize_uint8_linear(img, w, h, 0, resized.data(), new_w, new_h, 0, (stbir_pixel_layout)3);
    stbi_image_free(img);

    int crop_x = (new_w - target_size) / 2;
    int crop_y = (new_h - target_size) / 2;

    rgb_to_chw_float(resized.data(), new_w, crop_x, crop_y, target_size,
                     out, effective_mean, effective_std);
}

/* Write preprocessed image from memory directly into output buffer (zero-copy) */
inline void preprocess_image_bytes_to_buffer(
        const unsigned char* data, int data_size,
        float* out,
        int target_size = 224,
        const float mean[3] = nullptr,
        const float std_dev[3] = nullptr) {

    const float* effective_mean = mean ? mean : IMAGE_NET_MEAN;
    const float* effective_std = std_dev ? std_dev : IMAGE_NET_STD;

    int w, h, c;
    unsigned char* img = stbi_load_from_memory(data, data_size, &w, &h, &c, 3);
    if (!img) throw std::runtime_error("Failed to decode image from memory");
    c = 3;

    int new_w, new_h;
    if (w < h) { new_w = target_size; new_h = (int)((float)h / w * target_size); }
    else { new_h = target_size; new_w = (int)((float)w / h * target_size); }

    std::vector<unsigned char> resized(new_w * new_h * 3);
    stbir_resize_uint8_linear(img, w, h, 0, resized.data(), new_w, new_h, 0, (stbir_pixel_layout)3);
    stbi_image_free(img);

    int crop_x = (new_w - target_size) / 2;
    int crop_y = (new_h - target_size) / 2;

    rgb_to_chw_float(resized.data(), new_w, crop_x, crop_y, target_size,
                     out, effective_mean, effective_std);
}

}} /* namespace lembed::detail */

#endif /* LIBEMBEDDING_NO_IMAGE */
#endif /* LIBEMBEDDING_DETAIL_IMAGE_PREPROCESS_HPP */




