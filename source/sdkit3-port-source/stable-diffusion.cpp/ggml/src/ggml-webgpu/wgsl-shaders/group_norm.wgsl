struct Params {
    offset_src: u32,
    offset_dst: u32,

    ne0: u32,
    ne1: u32,
    ne2: u32,
    ne3: u32,
    n_groups: u32,
    eps: f32,
};

@group(0) @binding(0)
var<storage, read_write> src: array<f32>;

#ifdef INPLACE
@group(0) @binding(1)
var<uniform> params: Params;
#else
@group(0) @binding(1)
var<storage, read_write> dst: array<f32>;

@group(0) @binding(2)
var<uniform> params: Params;
#endif

var<workgroup> scratch: array<f32, WG_SIZE>;

fn src_index(index_in_group: u32, batch: u32, channel_start: u32) -> u32 {
    let spatial = params.ne0 * params.ne1;
    let channel = channel_start + index_in_group / spatial;
    let within_channel = index_in_group % spatial;
    return params.offset_src + batch * params.ne2 * spatial + channel * spatial + within_channel;
}

@compute @workgroup_size(WG_SIZE)
fn main(@builtin(workgroup_id) workgroup_id: vec3<u32>,
        @builtin(local_invocation_id) local_id: vec3<u32>) {
    let group = workgroup_id.x % params.n_groups;
    let batch = workgroup_id.x / params.n_groups;
    if (batch >= params.ne3) {
        return;
    }
    let channels_per_group = (params.ne2 + params.n_groups - 1u) / params.n_groups;
    let channel_start = group * channels_per_group;
    let group_channels = min(channels_per_group, params.ne2 - channel_start);
    let spatial = params.ne0 * params.ne1;
    let group_size = spatial * group_channels;

    var partial_sum = 0.0f;
    for (var base = 0u; base < group_size; base += WG_SIZE) {
        let i = base + local_id.x;
        if (i < group_size) {
            partial_sum += src[src_index(i, batch, channel_start)];
        }
    }
    scratch[local_id.x] = partial_sum;
    workgroupBarrier();

    var reduction_width = WG_SIZE / 2u;
    while (reduction_width > 0u) {
        if (local_id.x < reduction_width) {
            scratch[local_id.x] += scratch[local_id.x + reduction_width];
        }
        reduction_width /= 2u;
        workgroupBarrier();
    }
    let mean = scratch[0] / f32(group_size);
    // All lanes must read the reduced mean before any lane reuses scratch.
    workgroupBarrier();

    var partial_variance = 0.0f;
    for (var base = 0u; base < group_size; base += WG_SIZE) {
        let i = base + local_id.x;
        if (i < group_size) {
            let centered = src[src_index(i, batch, channel_start)] - mean;
            partial_variance += centered * centered;
        }
    }
    scratch[local_id.x] = partial_variance;
    workgroupBarrier();

    reduction_width = WG_SIZE / 2u;
    while (reduction_width > 0u) {
        if (local_id.x < reduction_width) {
            scratch[local_id.x] += scratch[local_id.x + reduction_width];
        }
        reduction_width /= 2u;
        workgroupBarrier();
    }
    let inverse_stddev = inverseSqrt(scratch[0] / f32(group_size) + params.eps);

    // Finish all reads for this group before writing; this also makes INPLACE safe.
    for (var base = 0u; base < group_size; base += WG_SIZE) {
        let i = base + local_id.x;
        if (i < group_size) {
            let input_index = src_index(i, batch, channel_start);
            let value = (src[input_index] - mean) * inverse_stddev;
#ifdef INPLACE
            src[input_index] = value;
#else
            dst[params.offset_dst + batch * params.ne2 * spatial + channel_start * spatial + i] = value;
#endif
        }
    }
}
