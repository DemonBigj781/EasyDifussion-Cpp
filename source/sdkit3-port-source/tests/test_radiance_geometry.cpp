#include "model/diffusion/flux.hpp"
#include "ggml-cpu.h"
#include <cstdio>
#include <stdexcept>
static void require(bool ok, const char* msg) { if (!ok) throw std::runtime_error(msg); }
    struct TinyApproximator : Flux::ChromaApproximator {
        TinyApproximator() : Flux::ChromaApproximator(64,8) {
            blocks.clear(); inner_size=8; n_layers=1;
            blocks["in_proj"]=std::make_shared<Linear>(64,8,true);
            blocks["norms.0"]=std::make_shared<::Flux::RMSNorm>(8);
            blocks["layers.0"]=std::make_shared<::Flux::MLPEmbedder>(8,8);
            blocks["out_proj"]=std::make_shared<Linear>(8,8,true);
        }
    };
    struct TinyModel : Flux::Flux {
        TinyModel(::Flux::FluxConfig config) : ::Flux::Flux(config) {
            if(config.is_chroma) blocks["distilled_guidance_layer"]=std::make_shared<TinyApproximator>();
        }
    };

static Flux::FluxConfig tiny_config(bool fake, bool pruned, bool x0) {
    Flux::FluxConfig cfg;
    cfg.version=VERSION_CHROMA_RADIANCE; cfg.patch_size=fake ? 32 : 16;
    cfg.in_channels=3; cfg.out_channels=3; cfg.hidden_size=8; cfg.num_heads=1;
    cfg.context_in_dim=4; cfg.vec_in_dim=0; cfg.guidance_embed=false;
    cfg.depth=1; cfg.depth_single_blocks=1; cfg.axes_dim={2,2,4}; cfg.axes_dim_sum=8;
    cfg.chroma_radiance_params.nerf_hidden_size=4;
    cfg.chroma_radiance_params.nerf_depth=1;
    cfg.chroma_radiance_params.nerf_max_freqs=2;
    cfg.chroma_radiance_params.nerf_mlp_ratio=2;
    cfg.chroma_radiance_params.fake_patch_size_x2=fake;
    cfg.chroma_radiance_params.use_x0=x0;
    cfg.is_chroma=pruned;
    return cfg;
}
struct TestWeights : RunnerWeightManager {
    bool fail=false;
    bool assign_compute_backend(const std::vector<ggml_tensor*>&,ggml_backend_t) override { return true; }
    bool prepare_params(const std::vector<ggml_tensor*>&) override { return !fail; }
    void release_compute_backend_params(const std::vector<ggml_tensor*>&) override {}
    void release_params_backend_params(const std::vector<ggml_tensor*>&) override {}
};
struct TinyRunner : Flux::FluxRunner {
    ggml_backend_buffer_t storage=nullptr;
    TinyRunner(ggml_backend_t cpu, const std::shared_ptr<TestWeights>& weights, bool fake, bool pruned, bool x0)
        : Flux::FluxRunner(cpu, {}, "", VERSION_CHROMA_RADIANCE, weights) {
        config=tiny_config(fake,pruned,x0); flux=TinyModel(config);
        free_params_ctx(); alloc_params_ctx(); flux.init(params_ctx);
        storage=ggml_backend_alloc_ctx_tensors(params_ctx,cpu);
        require(storage!=nullptr,"parameter allocation failed");
        for(auto t=ggml_get_first_tensor(params_ctx);t;t=ggml_get_next_tensor(params_ctx,t)) {
            std::vector<float> v(ggml_nelements(t)); for(size_t i=0;i<v.size();++i) v[i]=.01f+float(i%17)*.001f;
            if(t->type==GGML_TYPE_F32) ggml_backend_tensor_set(t,v.data(),0,ggml_nbytes(t));
            else if(t->type==GGML_TYPE_F16) { std::vector<ggml_fp16_t> f(v.size()); for(size_t i=0;i<v.size();++i) f[i]=ggml_fp32_to_fp16(v[i]); ggml_backend_tensor_set(t,f.data(),0,ggml_nbytes(t)); }
            else throw std::runtime_error("unexpected weight type");
        }
    }
    ~TinyRunner() { runner_done(); ggml_backend_buffer_free(storage); }
};
static void equal(const sd::Tensor<float>& a,const sd::Tensor<float>& b) {
    require(a.shape()==b.shape(),"shape changed");
    for(int64_t i=0;i<a.numel();++i) require(std::isfinite(a[i]) && a[i]==b[i],"numerical mismatch");
}

// Fresh reconstruction oracle: execute the actual final RMSNorm/3x3 convolution
// with one-hot neighbor kernels. Padded features deliberately differ from the
// requested seam, so a crop only after convolution cannot satisfy this oracle.
static void check_final_convolution_seams(ggml_backend_t cpu) {
    int cases = 0;
    for (auto shape : {std::pair<int,int>{5,5}, {8,5}, {5,8}, {7,6}, {8,8}}) {
        const int w=shape.first, h=shape.second, pw=8, ph=8;
        for (int axes=0; axes<4; ++axes) for (int dy=-1; dy<=1; ++dy) for (int dx=-1; dx<=1; ++dx) {
            auto ctx=std::unique_ptr<ggml_context,decltype(&ggml_free)>(ggml_init({4*1024*1024,nullptr,false}),ggml_free);
            require(ctx != nullptr,"seam context allocation failed");
            Flux::NerfFinalLayerConv layer(2,1);
            layer.init(ctx.get());
            std::map<std::string,ggml_tensor*> params;
            layer.get_param_tensors(params, "final");
            for (auto& [name,tensor] : params) ggml_set_zero(tensor);
            ggml_set_f32(params.at("final.norm.scale"),1.f);
            ggml_set_f32_1d(params.at("final.conv.weight"),(dy+1)*3+dx+1,1.f);
            auto input=ggml_new_tensor_4d(ctx.get(),GGML_TYPE_F32,pw,ph,2,1);
            for(int y=0;y<ph;++y) for(int x=0;x<pw;++x) {
                // Binary-exact normalized values after F16 convolution casting.
                ggml_set_f32_1d(input,y*pw+x,static_cast<float>((x+2*y)%7-3));
                ggml_set_f32_1d(input,pw*ph+y*pw+x,2.f);
            }
            GGMLRunnerContext runner;
            runner.ggml_ctx=ctx.get(); runner.backend=cpu;
            runner.circular_x_enabled=axes&1; runner.circular_y_enabled=axes&2;
            const int cw=(axes&1)?w:pw, ch=(axes&2)?h:ph;
            auto cropped=Flux::radiance_crop_spatial(ctx.get(),input,cw,ch);
            require((cropped==input)==(cw==pw && ch==ph),"unnecessary or missing convolution crop");
            auto output=layer.forward(&runner,cropped);
            output=Flux::radiance_crop_spatial(ctx.get(),output,w,h);
            auto graph=ggml_new_graph(ctx.get()); ggml_build_forward_expand(graph,output);
            require(ggml_graph_compute_with_ctx(ctx.get(),graph,1)==GGML_STATUS_SUCCESS,"seam graph failed");
            for(int y=0;y<h;++y) for(int x=0;x<w;++x) {
                int sx=x+dx, sy=y+dy;
                if(axes&1) sx=(sx+w)%w;
                if(axes&2) sy=(sy+h)%h;
                float expected=0.f;
                if(sx>=0 && sx<cw && sy>=0 && sy<ch) {
                    float v=static_cast<float>((sx+2*sy)%7-3);
                    expected=v/std::sqrt((v*v+4.f)*.5f+1e-6f);
                    expected=ggml_fp16_to_fp32(ggml_fp32_to_fp16(expected));
                }
                require(std::abs(ggml_get_f32_1d(output,y*w+x)-expected)<0.001f,
                        "final convolution wrapped at padded rather than requested seam");
            }
            ++cases;
        }
    }
    std::printf("PASS %d final-convolution neighbor-selection cases\n",cases);
}

int main() {
    auto cpu=std::unique_ptr<ggml_backend,decltype(&ggml_backend_free)>(ggml_backend_cpu_init(),ggml_backend_free);
    ggml_backend_cpu_set_n_threads(cpu.get(),1);
    try { check_final_convolution_seams(cpu.get()); int cases=0;
        for(bool pruned:{false,true}) for(bool fake:{false,true}) {
            auto weights=std::make_shared<TestWeights>(); TinyRunner runner(cpu.get(),weights,fake,pruned,false);
            for(auto shape:{std::pair<int,int>{64,64},{96,64},{65,65},{65,96},{96,65},{63,63}}) for(int circular=0;circular<4;++circular) {
                int w=shape.first,h=shape.second; runner.set_circular_axes(circular&1,circular&2);
                sd::Tensor<float> x({w,h,3,1}), text({4,2,1}), t({1}), guidance({1});
                for(int64_t i=0;i<x.numel();++i) x[i]=float(i%31)/31.f;
                for(int64_t i=0;i<text.numel();++i) text[i]=float(i+1)*.1f;
                t[0]=.5f; guidance[0]=0.f;
                auto run=[&]() { return runner.compute(1,x,t,text,{}, {},guidance); };
                sd::Tensor<float> normal;
                for(bool x0:{false,true}) {
                    runner.config.chroma_radiance_params.use_x0=runner.flux.config.chroma_radiance_params.use_x0=x0;
                    auto baseline=run(); runner.runner_done();
                    require(!baseline.empty() && baseline.shape()==x.shape(),"candidate shape/empty failure");
                    require(runner.dct_vec==runner.fetch_dct_pos(runner.config.patch_size,2),"actual DCT not retained");
                    if(!x0) normal=baseline;
                    else for(int64_t i=0;i<x.numel();++i) require(std::abs(baseline[i]-(x[i]-normal[i])*2.f)<1e-6f,"x0 crop mismatch");
                    weights->fail=true; require(run().empty(),"weight preparation failure ignored"); runner.runner_done(); weights->fail=false;
                    equal(baseline,run()); runner.runner_done();
                    for(int stop:{1,2}) {
                        int calls=0; runner.set_execution_cancel_callback([&]() { return ++calls>=stop; });
                        require(run().empty(),"cancellation ignored"); require(calls>=stop,"cancellation boundary not reached"); runner.runner_done();
                        runner.set_execution_cancel_callback({}); equal(baseline,run()); runner.runner_done();
                    }
                    ++cases;
                }
                std::printf("PASS runner %dx%d fake=%d pruned=%d circular=%d x0 both\n",w,h,fake,pruned,circular);
            }
        }
        std::printf("PASS %d runner cases including DCT, weight failure, cancellation and recovery\n",cases);
    } catch(const std::exception& e) { std::fprintf(stderr,"%s\n",e.what()); return 1; }
}
