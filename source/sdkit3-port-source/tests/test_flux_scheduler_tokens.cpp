#include "stable-diffusion.cpp"
#include "ggml-cpu.h"
#include <cstdio>
#include <stdexcept>
static void require(bool ok, const char* message) { if (!ok) throw std::runtime_error(message); }
struct DefaultPatchRunner : DiffusionModelRunner {
    explicit DefaultPatchRunner(ggml_backend_t cpu) : DiffusionModelRunner(cpu, "test") {}
    std::string get_desc() override { return "test"; }
    sd::Tensor<float> compute(int, const DiffusionParams&) override { return {}; }
    void get_param_tensors(std::map<std::string,ggml_tensor*>&,const std::string&) override {}
};
struct RecordingFluxDenoiser : FluxFlowDenoiser {
    int calls=0, sequence=0;
    std::vector<float> get_sigmas(uint32_t n,int length,scheduler_t scheduler,SDVersion version,const char* args=nullptr) override {
        ++calls; sequence=length;
        return FluxFlowDenoiser::get_sigmas(n,length,scheduler,version,args);
    }
};
static void check_sigmas(const std::vector<float>& sigmas,int tokens,int steps) {
    require(sigmas.size() == static_cast<size_t>(steps+1),"sigma count changed");
    double mu=0.5+(tokens-256.0)*(1.15-0.5)/(4096.0-256.0);
    for(int i=0;i<=steps;++i) {
        double t=1.0-static_cast<double>(i)/steps;
        double expected=t == 0 ? 0 : std::exp(mu)/(std::exp(mu)+(1.0/t-1.0));
        require(std::abs(sigmas[i]-expected) < 1e-6,"Flux sigma differs from analytical patch-token schedule");
    }
}
int main() {
    auto cpu=std::unique_ptr<ggml_backend,decltype(&ggml_backend_free)>(ggml_backend_cpu_init(),ggml_backend_free);
    try {
        require(cpu != nullptr,"CPU unavailable");
        StableDiffusionGGML app;
        app.version=VERSION_FLUX;
        app.first_stage_model=std::make_shared<FakeVAE>(VERSION_FLUX,cpu.get());
        auto flux=std::make_shared<Flux::FluxRunner>(cpu.get());
        app.diffusion_model=flux;
        require(app.get_image_seq_len(1024,1024) == 4096,"Flux scheduler used latent pixels instead of patch tokens");
        for(auto version : {VERSION_PID,VERSION_FLUX,VERSION_FLUX2,VERSION_LTXV}) {
            app.version=version;
            app.first_stage_model=std::make_shared<FakeVAE>(version,cpu.get());
            int scale=app.get_vae_scale_factor();
            for(int patch : {1,2,16,32}) {
                flux->config.patch_size=patch;
                for(int h : {1,7,8,15,16,31,32,63,64,65,127,128,255,256,511,512,513,1024,1536})
                for(int w : {1,8,16,31,64,127,256,513,1024}) {
                    int latent_h=h/scale, latent_w=w/scale;
                    int expected=(latent_h/patch+(latent_h%patch != 0))*(latent_w/patch+(latent_w%patch != 0));
                    require(app.get_image_seq_len(h,w) == expected,"patch token count differs for odd or rectangular dimensions");
                    if(latent_h > 0 && latent_w > 0) {
                        auto ctx=std::unique_ptr<ggml_context,decltype(&ggml_free)>(ggml_init({1024*1024,nullptr,true}),ggml_free);
                        GGMLRunnerContext runner_ctx; runner_ctx.ggml_ctx=ctx.get(); runner_ctx.backend=cpu.get();
                        auto input=ggml_new_tensor_4d(ctx.get(),GGML_TYPE_F32,latent_w,latent_h,3,1);
                        auto patches=DiT::pad_and_patchify(&runner_ctx,input,patch,patch);
                        require(patches->ne[1] == app.get_image_seq_len(h,w),"scheduler length differs from actual patchify graph shape");
                    }
                }
            }
            app.diffusion_model=std::make_shared<DefaultPatchRunner>(cpu.get());
            require(app.get_image_seq_len(1025,513) == (1025/scale)*(513/scale),"default runner behavior changed");
            app.diffusion_model.reset();
            require(app.get_image_seq_len(1025,513) == (1025/scale)*(513/scale),"missing runner behavior changed");
            app.diffusion_model=flux;
        }
        app.version=VERSION_FLUX; app.first_stage_model=std::make_shared<FakeVAE>(VERSION_FLUX,cpu.get());
        flux->config.patch_size=2; app.diffusion_model=flux;
        auto denoiser=std::make_shared<RecordingFluxDenoiser>(); app.denoiser=denoiser;
        sd_ctx_t context{&app};
        for(int size : {512,1024,1536}) {
            sd_img_gen_params_t params; sd_img_gen_params_init(&params);
            params.width=size; params.height=1024;
            params.sample_params.sample_steps=4;
            params.sample_params.scheduler=FLUX_SCHEDULER;
            params.sample_params.sample_method=EULER_SAMPLE_METHOD;
            GenerationRequest request(&context,&params);
            int tokens=(request.width/16)*(request.height/16);
            SamplePlan plan(&context,&params,request);
            require(denoiser->sequence == tokens,"initial SamplePlan received wrong token count");
            check_sigmas(plan.sigmas,tokens,4);
            sd_hires_params_t hires{}; hires.steps=4; hires.denoising_strength=1.f;
            int steps=0;
            auto sigma=make_hires_sigma_schedule(&context,hires,params.sample_params,EULER_SAMPLE_METHOD,4,
                                                 app.get_image_seq_len(1536,size),&steps);
            check_sigmas(sigma,(1536/16)*(size/16),steps);
            float custom[]={0.8f,0.4f,0.f};
            params.sample_params.custom_sigmas=custom; params.sample_params.custom_sigmas_count=3;
            int calls=denoiser->calls;
            SamplePlan custom_plan(&context,&params,request);
            require(denoiser->calls == calls && custom_plan.sigmas == std::vector<float>({0.8f,0.4f,0.f}),"custom sigma bypass changed");
            hires.custom_sigmas=custom; hires.custom_sigmas_count=3;
            auto custom_hires=make_hires_sigma_schedule(&context,hires,params.sample_params,EULER_SAMPLE_METHOD,4,1,&steps);
            require(denoiser->calls == calls && custom_hires == custom_plan.sigmas,"hires custom sigma bypass changed");
        }
        // Preserve and document the inherited fake-x2 Radiance shape discrepancy.
        for(int size : {64,65}) {
            auto ctx=std::unique_ptr<ggml_context,decltype(&ggml_free)>(ggml_init({1024*1024,nullptr,true}),ggml_free);
            GGMLRunnerContext runner_ctx; runner_ctx.ggml_ctx=ctx.get(); runner_ctx.backend=cpu.get();
            auto input=ggml_new_tensor_4d(ctx.get(),GGML_TYPE_F32,size,size,3,1);
            auto padded=DiT::pad_to_patch_size(&runner_ctx,input,32,32);
            auto original_patches=DiT::patchify(ctx.get(),padded,32,32);
            auto resized=ggml_interpolate(ctx.get(),padded,size/2,size/2,3,1,GGML_SCALE_MODE_BILINEAR);
            auto kernel=ggml_new_tensor_4d(ctx.get(),GGML_TYPE_F16,16,16,3,4);
            auto projected=ggml_conv_2d(ctx.get(),kernel,resized,16,16,0,0,1,1);
            int64_t projected_tokens=projected->ne[0]*projected->ne[1];
            require(projected_tokens == 4 && original_patches->ne[1] == (size == 64 ? 4 : 9),"Radiance audit reproduction changed");
        }
        std::puts("PASS actual scheduler sequence helper, Flux patch sizes, VAE scales, rectangular/odd dimensions and default fallback");
    } catch(const std::exception& e) { std::fprintf(stderr,"%s\n",e.what()); return 1; }
}
