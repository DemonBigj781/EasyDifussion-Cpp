"""Color guidance stays separate from the original image and ordinary mask."""
import ast
import importlib.util
from pathlib import Path
import queue
import sys
import types
import unittest
from unittest import mock

from PIL import Image

ROOT = Path(__file__).resolve().parents[1]


def load_module(relative, name):
    spec = importlib.util.spec_from_file_location(name, ROOT / relative)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


class ColorWeightedInpaintTests(unittest.TestCase):
    def prepare(self, *args, **kwargs):
        helper = load_module('ui/easydiffusion/color_inpaint.py', 'color_inpaint_test')
        return helper.prepare_color_weighted_inpaint(*args, **kwargs)

    def test_independent_mask_then_weighted_color_in_one_conditioning_input(self):
        original = Image.new('RGB', (4, 1), 'white')
        guide = Image.new('RGBA', (4, 1))
        guide.putpixel((0, 0), (255, 0, 0, 64))
        guide.putpixel((1, 0), (0, 0, 255, 192))
        mask = Image.new('L', (4, 1)); mask.putpixel((2, 0), 128)
        before = [item.tobytes() for item in (original, guide, mask)]
        conditioned, effective_mask = self.prepare(original, guide, mask, (4, 1), .65)
        self.assertEqual(conditioned.getpixel((0, 0)), (255, 191, 191))
        self.assertEqual(conditioned.getpixel((1, 0)), (63, 63, 255))
        self.assertEqual(conditioned.getpixel((2, 0)), (255, 255, 255))
        self.assertEqual(list(effective_mask.convert('L').getdata()), [255, 255, 128, 0])
        self.assertEqual([item.tobytes() for item in (original, guide, mask)], before)

    def test_advanced_can_operate_without_an_ordinary_mask(self):
        source = Image.new('RGB', (2, 1), 'white')
        guide = Image.new('RGBA', (2, 1)); guide.putpixel((0, 0), (0, 0, 0, 128))
        conditioned, mask = self.prepare(source, guide, None, (2, 1), .65)
        self.assertEqual(conditioned.getpixel((0, 0)), (127, 127, 127))
        self.assertEqual(list(mask.convert('L').getdata()), [255, 0])

    def test_invalid_or_empty_guidance_is_not_silently_treated_as_drawing(self):
        source = Image.new('RGB', (2, 2), 'white')
        for guide in (Image.new('RGB', (2, 2)), Image.new('RGBA', (3, 3)), Image.new('RGBA', (2, 2))):
            with self.subTest(mode=guide.mode, size=guide.size):
                with self.assertRaises(ValueError): self.prepare(source, guide, None, source.size, .65)
        with self.assertRaisesRegex(ValueError, 'strength'):
            self.prepare(source, Image.new('RGBA', (2, 2), (255, 0, 0, 128)), None, source.size, 1)

    def test_request_schema_keeps_the_independent_color_layer(self):
        schema = load_module('ui/easydiffusion/types.py', 'color_request_schema')
        request = schema.GenerateImageRequest(init_image='original', init_image_mask='ordinary', inpaint_color_guide='rgba-guide')
        self.assertEqual(request.inpaint_color_guide, 'rgba-guide')
        self.assertEqual(request.init_image_mask, 'ordinary')

    def test_render_api_preserves_both_mask_names_and_color_guidance(self):
        schema = load_module('ui/easydiffusion/types.py', 'color_api_schema')
        source = ROOT / 'ui/easydiffusion/server.py'
        function = next(node for node in ast.parse(source.read_text()).body if isinstance(node,ast.FunctionDef) and node.name=='render_internal')
        namespace={name:getattr(schema,name) for name in ('GenerateImageRequest','RenderTaskData','ModelsData','OutputFormatData','SaveToDiskData','convert_legacy_render_req_to_new')}
        class HttpError(Exception):
            def __init__(self,status_code,detail):
                super().__init__(detail);self.status_code=status_code
        namespace.update(kiosk=types.SimpleNamespace(validate_render=lambda req:None),
                         app=types.SimpleNamespace(getConfig=lambda:{},save_to_config=mock.Mock()),
                         RenderTask=lambda req,*args:req,enqueue_task=lambda task:task,HTTPException=HttpError)
        exec(compile(ast.Module(body=[function],type_ignores=[]),str(source),'exec'),namespace)
        for fields, expected in (({'mask':'legacy-mask'},'legacy-mask'),
                                 ({'init_image_mask':'native-mask'},'native-mask'),
                                 ({'init_image_mask':None,'mask':'old-mask'},None)):
            with self.subTest(fields=fields):
                request=namespace['render_internal']({'init_image':'original','inpaint_color_guide':'rgba-guide',**fields})
                self.assertEqual(request.init_image_mask,expected)
                self.assertEqual(request.inpaint_color_guide,'rgba-guide')
        for strength in (0,1):
            with self.subTest(strength=strength), self.assertRaises(HttpError) as error:
                namespace['render_internal']({'init_image':'original','inpaint_color_guide':'rgba-guide','prompt_strength':strength})
            self.assertEqual(error.exception.status_code,400)

    def test_both_task_paths_make_one_backend_call_without_mutating_original(self):
        helper = load_module('ui/easydiffusion/color_inpaint.py', 'color_inpaint_adapter_test')
        for relative in ('ui/easydiffusion/tasks/render_images.py', 'ui/plugins/server/tasks/render_images.py'):
            with self.subTest(source=relative):
                function = next(node for node in ast.parse((ROOT / relative).read_text()).body
                                if isinstance(node, ast.FunctionDef) and node.name == 'generate_images_internal')
                backend = types.SimpleNamespace(set_options=mock.Mock(), generate_images=mock.Mock(return_value=[]))
                module = types.ModuleType('easydiffusion.backend_manager'); module.backend = backend
                namespace = {name: object for name in ('GenerateImageRequest','RenderTaskData','ModelsData','OutputFormatData')}
                namespace.update(queue=queue, make_step_callback=lambda *args:None, get_image=lambda value:value,
                                 resize_img=lambda image,w,h,**kwargs:image.resize((w,h)),
                                 prepare_color_weighted_inpaint=helper.prepare_color_weighted_inpaint,
                                 combine_positive_prompt_for_clip=lambda prompt,hidden:prompt,
                                 combine_negative_prompt_for_clip=lambda prompt,hidden:prompt)
                exec(compile(ast.Module(body=[function],type_ignores=[]), relative, 'exec'),namespace)
                class Request(types.SimpleNamespace):
                    def dict(self): return dict(vars(self))
                source = Image.new('RGB',(8,8),'white')
                guide = Image.new('RGBA',(8,8)); guide.putpixel((0,0),(255,0,0,64))
                normal = Image.new('RGB',(8,8)); normal.putpixel((7,7),(128,128,128))
                request=Request(width=8,height=8,control_image=None,init_image=source,init_image_mask=normal,
                                inpaint_color_guide=guide,num_inference_steps=8,prompt_strength=.65,
                                prompt='test',negative_prompt='',hidden_positive_prompt='',hidden_negative_prompt='')
                task=types.SimpleNamespace(control_filter_to_apply=None,control_image_preprocessed=False,enable_vae_tiling=False,clip_skip=False)
                output=types.SimpleNamespace(output_format='png',output_quality=90,output_lossless=True)
                with mock.patch.dict(sys.modules,{'easydiffusion.backend_manager':module}):
                    namespace['generate_images_internal'](None,request,task,None,output,None,[],None,False,1)
                self.assertEqual(backend.generate_images.call_count,1)
                payload=backend.generate_images.call_args.kwargs
                self.assertNotIn('inpaint_color_guide',payload)
                self.assertEqual(payload['init_image'].getpixel((0,0)),(255,191,191))
                self.assertEqual(payload['init_image_mask'].getpixel((7,7)),(128,128,128))
                self.assertEqual(source.getpixel((0,0)),(255,255,255))
                self.assertIs(request.init_image,source)


if __name__ == '__main__': unittest.main()
