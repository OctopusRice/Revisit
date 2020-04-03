// modify from
// https://github.com/chengdazhi/Deformable-Convolution-V2-PyTorch/blob/mmdetection/mmdet/ops/dcn/src/modulated_dcn_cuda.c

// based on
// author: Charles Shang
// https://github.com/torch/cunn/blob/master/lib/THCUNN/generic/SpatialConvolutionMM.cu

#include <torch/extension.h>
#include <ATen/DeviceGuard.h>

#include <cmath>
#include <vector>

void DeformablePSROIPoolForward(
    const at::Tensor data, const at::Tensor bbox, const at::Tensor trans,
    at::Tensor out, at::Tensor top_count, const int batch, const int channels,
    const int height, const int width, const int num_bbox,
    const int channels_trans, const int no_trans, const float spatial_scale,
    const int output_dim, const int group_size, const int pooled_size,
    const int part_size, const int sample_per_part, const float trans_std);

void DeformablePSROIPoolBackwardAcc(
    const at::Tensor out_grad, const at::Tensor data, const at::Tensor bbox,
    const at::Tensor trans, const at::Tensor top_count, at::Tensor in_grad,
    at::Tensor trans_grad, const int batch, const int channels,
    const int height, const int width, const int num_bbox,
    const int channels_trans, const int no_trans, const float spatial_scale,
    const int output_dim, const int group_size, const int pooled_size,
    const int part_size, const int sample_per_part, const float trans_std);

void deform_psroi_pooling_cuda_forward(
    at::Tensor input, at::Tensor bbox, at::Tensor trans, at::Tensor out,
    at::Tensor top_count, const int no_trans, const float spatial_scale,
    const int output_dim, const int group_size, const int pooled_size,
    const int part_size, const int sample_per_part, const float trans_std);

void deform_psroi_pooling_cuda_backward(
    at::Tensor out_grad, at::Tensor input, at::Tensor bbox, at::Tensor trans,
    at::Tensor top_count, at::Tensor input_grad, at::Tensor trans_grad,
    const int no_trans, const float spatial_scale, const int output_dim,
    const int group_size, const int pooled_size, const int part_size,
    const int sample_per_part, const float trans_std);
