#!/usr/bin/env python3
"""Real Chromium image-input checks against rendered C++ HTML and production JS.

Usage: python verify_img2img_browser.py --renderer /path/easy-diffusion-ui-render
Requires Playwright plus Chromium. API responses are fixtures; no inference/GPU.
"""
import argparse
import base64
import io
import json
from pathlib import Path
import re
import subprocess
from PIL import Image
from playwright.sync_api import sync_playwright

ROOT = Path(__file__).resolve().parents[2]
parser = argparse.ArgumentParser()
parser.add_argument('--renderer', required=True)
parser.add_argument('--chromium', default='/usr/bin/chromium')
args = parser.parse_args()
html = subprocess.check_output([args.renderer, '/'], text=True)
html = re.sub(r'<script\b[^>]*>.*?</script>', '', html, flags=re.S)
script = (ROOT / 'cosmopolitan/resources/cpp-ui/scripts/generate.js').read_text()
buffer = io.BytesIO()
Image.new('RGB', (8, 8), (32, 80, 160)).save(buffer, format='PNG')
png = buffer.getvalue()
b64 = base64.b64encode(png).decode()
bootstrap = r'''
window.CppKiosk = {ready: Promise.resolve({capabilities:{txt2img:true,img2img_ui:true,uploaded_mask_inpainting_ui:true,advanced_inpaint_ui:true,max_concurrent_generations:1}})};
window.calls=[];window.progressState={current_step:0,total_steps:4,completed:false};
window.fetch=async(path,options={})=>{
 const body=options.body?JSON.parse(options.body):undefined; calls.push({path,body});
 let value={};
 if(path.endsWith('/backend-devices'))value={devices:[{backend:'cpu',selector:'CPU'}]};
 else if(path.endsWith('/config'))value={effective:{compute:{backend:'cpu',device:'auto'},inference:{image:{prompt:'cube',negative_prompt:'',width:128,height:128,steps:4,cfg_scale:7,seed:42,sampler_name:'euler',scheduler:'discrete'}}}};
 else if(path.endsWith('/checkpoints'))value={models:[{name:'fixture.safetensors'}]};
 else if(path.endsWith('/options'))value={sd_model_checkpoint:'fixture.safetensors'};
 else if(path.endsWith('/progress'))value=progressState;
 else if(path.endsWith('/settings'))value={saved:body};
 else if(path.endsWith('/img2img')||path.endsWith('/txt2img'))return new Promise(resolve=>window.finishGeneration=value=>resolve(new Response(JSON.stringify(value))));
 else if(!path.endsWith('/interrupt'))throw Error('Unexpected fixture API: '+path);
 return new Response(JSON.stringify(value));
};
'''
with sync_playwright() as p:
    browser = p.chromium.launch(executable_path=args.chromium, headless=True, args=['--disable-gpu'])
    page = browser.new_page(viewport={'width':1280,'height':900})
    errors = []
    page.on('pageerror', lambda error: errors.append(str(error)))
    page.route('**/*', lambda route: route.fulfill(status=200, content_type='text/html', body=html)
               if route.request.resource_type == 'document' else route.abort())
    page.goto('http://native-ui.test/')
    page.add_script_tag(content=bootstrap)
    page.add_script_tag(content=script)
    page.wait_for_function('!document.getElementById("makeImage").disabled')
    upload = {'name':'initial.png','mimeType':'image/png','buffer':png}
    def put(selector, file=upload):
        page.locator(selector).set_input_files(file)
    def canvas_box(selector):
        # Raw page.mouse coordinates do not auto-scroll like locator.click().
        # Editor controls and later control interactions can move the canvas
        # outside the viewport, so resolve fresh visible coordinates each time.
        canvas = page.locator(selector)
        canvas.scroll_into_view_if_needed()
        box = canvas.bounding_box()
        assert box and box['width'] > 0 and box['height'] > 0, selector
        for fraction in (.25, .5):
            point = {'x': box['x'] + box['width'] * fraction,
                     'y': box['y'] + box['height'] * fraction}
            assert canvas.evaluate('(c,p)=>document.elementFromPoint(p.x,p.y)===c', point), (selector, box)
        return box
    def ready(selector):
        page.wait_for_function('(id)=>{const e=document.querySelector(id);return !e.hidden && e.naturalWidth===8;}', arg=selector)
    put('#cosmo-initial-file'); ready('#cosmo-initial-preview')
    assert not page.evaluate('calls.some(c=>c.path.endsWith("img2img"))'), 'upload transmitted before Generate'
    put('#cosmo-mask-file'); ready('#cosmo-mask-preview')
    page.locator('#cosmo-image-strength').fill('0.45')
    page.locator('#cosmo-mask-blur').fill('2')
    page.locator('#cosmo-mask-invert').check()
    page.locator('#makeImage').click()
    page.wait_for_function('calls.some(c=>c.path.endsWith("img2img"))')
    call = page.evaluate('calls.find(c=>c.path.endsWith("img2img"))')
    assert call['body']['init_images'] == [b64] and call['body']['mask'] == b64
    assert call['body']['denoising_strength'] == .45 and call['body']['mask_blur'] == 2
    assert call['body']['inpainting_mask_invert'] == 1
    assert page.locator('#cosmo-initial-file').is_disabled()
    assert page.locator('#cosmo-mask-clear').is_disabled()
    page.evaluate('progressState.current_step=1')
    page.wait_for_function('!document.getElementById("stopImage").disabled')
    page.locator('#stopImage').click()
    assert page.evaluate('calls.find(c=>c.path.endsWith("interrupt")).body.id_task') == call['body']['force_task_id']
    page.evaluate('finishGeneration({images:[]})')
    page.wait_for_function('!document.getElementById("makeImage").disabled')
    # Clearing the source also clears its mask; repeated same-file upload works.
    page.locator('#cosmo-initial-clear').click()
    assert page.locator('#cosmo-mask-preview').is_hidden()
    assert page.locator('#cosmo-mask-file').is_disabled()
    put('#cosmo-initial-file'); ready('#cosmo-initial-preview')
    put('#cosmo-initial-file'); ready('#cosmo-initial-preview')
    page.locator('#makeImage').click()
    page.wait_for_function('calls.filter(c=>c.path.endsWith("img2img")).length===2')
    assert 'mask' not in page.evaluate('calls.filter(c=>c.path.endsWith("img2img"))[1].body')
    page.evaluate('finishGeneration({images:[]})')
    page.wait_for_function('!document.getElementById("makeImage").disabled')
    # Corrupt bytes replace, rather than retaining, the prior source.
    put('#cosmo-initial-file', {'name':'broken.png','mimeType':'image/png','buffer':b'not a png'})
    page.wait_for_function('document.getElementById("generation-queue-status").textContent.includes("not a decodable")')
    assert page.locator('#cosmo-initial-preview').is_hidden()
    page.locator('#makeImage').click()
    page.wait_for_function('calls.some(c=>c.path.endsWith("txt2img"))')
    assert 'init_images' not in page.evaluate('calls.find(c=>c.path.endsWith("txt2img")).body')
    page.evaluate('finishGeneration({images:[]})')
    page.wait_for_function('!document.getElementById("makeImage").disabled')
    # Oversized file and invalid strength do not start requests.
    put('#cosmo-initial-file', {'name':'huge.png','mimeType':'image/png','buffer':b'x'*(16*1024*1024+1)})
    page.wait_for_function('document.getElementById("generation-queue-status").textContent.includes("16 MiB")')
    put('#cosmo-initial-file'); ready('#cosmo-initial-preview')
    count = page.evaluate('calls.length')
    page.locator('#cosmo-image-strength').fill('2')
    page.locator('#makeImage').click()
    assert page.evaluate('calls.length') == count
    # Independent Advanced Inpaint inputs; real browser PNG and canvas decoding.
    def png_file(mode, size, color):
        stream = io.BytesIO(); Image.new(mode, size, color).save(stream, format='PNG')
        return {'name':'guide.png','mimeType':'image/png','buffer':stream.getvalue()}
    guide = png_file('RGBA',(8,8),(30,90,220,128))
    put('#cosmo-mask-file'); ready('#cosmo-mask-preview')
    put('#cosmo-guide-file',guide); ready('#cosmo-guide-preview')
    original = page.locator('#cosmo-initial-preview').get_attribute('src')
    guide_before = page.locator('#cosmo-guide-preview').get_attribute('src')
    page.locator('#cosmo-guide-edit').click()
    assert page.locator('#makeImage').is_disabled()
    page.wait_for_function('!document.getElementById("cosmo-guide-save").disabled')
    before = page.locator('#cosmo-guide-canvas').evaluate('(c)=>c.toDataURL()')
    box = canvas_box('#cosmo-guide-canvas')
    page.mouse.move(box['x']+box['width']/4,box['y']+box['height']/4)
    page.mouse.down(); page.mouse.move(box['x']+box['width']/2,box['y']+box['height']/2); page.mouse.up()
    assert page.locator('#cosmo-guide-canvas').evaluate('(c)=>c.toDataURL()') != before
    page.locator('#cosmo-guide-undo').click()
    assert page.locator('#cosmo-guide-canvas').evaluate('(c)=>c.toDataURL()') == before
    # Guide wand samples the uniform original, not the colored guide.
    page.locator('#cosmo-guide-tool').select_option('wand')
    page.locator('#cosmo-guide-sensitivity').fill('0')
    page.locator('#cosmo-guide-buffer').fill('0')
    box = canvas_box('#cosmo-guide-canvas')
    page.mouse.click(box['x']+box['width']/2,box['y']+box['height']/2)
    assert page.locator('#cosmo-guide-canvas').evaluate('(c)=>c.toDataURL()') != before
    assert page.locator('#cosmo-initial-preview').get_attribute('src') == original
    page.locator('#cosmo-guide-undo').click()
    assert page.locator('#cosmo-guide-canvas').evaluate('(c)=>c.toDataURL()') == before
    page.locator('#cosmo-guide-tool').select_option('brush')
    page.locator('#cosmo-guide-cancel').click()
    assert page.locator('#cosmo-guide-preview').get_attribute('src') == guide_before
    assert page.locator('#cosmo-initial-preview').get_attribute('src') == original
    page.locator('#cosmo-guide-edit').click()
    page.wait_for_function('!document.getElementById("cosmo-guide-save").disabled')
    page.locator('#cosmo-guide-save').click(); ready('#cosmo-guide-preview')
    saved = Image.open(io.BytesIO(base64.b64decode(page.locator('#cosmo-guide-preview').get_attribute('src').split(',')[1])))
    assert saved.size == (8,8) and saved.mode == 'RGBA'
    assert page.locator('#cosmo-initial-preview').get_attribute('src') == original
    page.locator('#cosmo-image-strength').fill('1')
    count = page.evaluate('calls.length');page.locator('#makeImage').click()
    assert page.evaluate('calls.length') == count
    assert 'strictly between' in page.locator('#generation-queue-status').inner_text()
    page.locator('#cosmo-image-strength').fill('.75')
    page.locator('#cosmo-mask-enabled').uncheck()
    page.locator('#makeImage').click()
    page.wait_for_function('calls.filter(c=>c.path.endsWith("img2img")).length===3')
    call = page.evaluate('calls.filter(c=>c.path.endsWith("img2img")).at(-1)')
    assert 'inpaint_color_guide' in call['body'] and 'mask' not in call['body']
    assert page.locator('#cosmo-guide-edit').is_disabled()
    page.evaluate('finishGeneration({images:[]})');page.wait_for_function('!document.getElementById("makeImage").disabled')
    for file, error in [(png_file('RGB',(8,8),(1,2,3)),'alpha channel'),(png_file('RGBA',(8,8),(1,2,3,0)),'no painted pixels'),(png_file('RGBA',(16,8),(1,2,3,128)),'matching dimensions')]:
        put('#cosmo-guide-file',file)
        page.wait_for_function('(text)=>document.getElementById("generation-queue-status").textContent.includes(text)',arg=error)
        assert page.locator('#cosmo-guide-preview').is_hidden()
    put('#cosmo-guide-file',guide); ready('#cosmo-guide-preview')
    # Ordinary editor exports actual opaque grayscale, never overlay alpha.
    guide_unchanged = page.locator('#cosmo-guide-preview').get_attribute('src')
    source_unchanged = page.locator('#cosmo-initial-preview').get_attribute('src')
    put('#cosmo-mask-file',png_file('RGB',(8,8),(0,0,0)))
    page.locator('#cosmo-mask-edit').click()
    page.wait_for_function('!document.getElementById("cosmo-mask-save").disabled')
    assert page.locator('#cosmo-guide-edit').is_disabled()
    page.locator('#cosmo-mask-reset').click();page.locator('#cosmo-mask-save').click();ready('#cosmo-mask-preview')
    saved_mask = Image.open(io.BytesIO(base64.b64decode(page.locator('#cosmo-mask-preview').get_attribute('src').split(',')[1]))).convert('RGBA')
    assert set(saved_mask.getdata()) == {(0,0,0,255)} and saved_mask.size == (8,8)
    page.locator('#cosmo-mask-edit').click();page.wait_for_function('!document.getElementById("cosmo-mask-save").disabled')
    page.locator('#cosmo-mask-tool').select_option('wand');page.locator('#cosmo-mask-sensitivity').fill('0')
    box = canvas_box('#cosmo-mask-canvas');page.mouse.click(box['x']+box['width']/2,box['y']+box['height']/2)
    page.locator('#cosmo-mask-save').click();ready('#cosmo-mask-preview')
    saved_mask = Image.open(io.BytesIO(base64.b64decode(page.locator('#cosmo-mask-preview').get_attribute('src').split(',')[1]))).convert('RGBA')
    assert set(saved_mask.getdata()) == {(255,255,255,255)}
    assert page.locator('#cosmo-guide-preview').get_attribute('src') == guide_unchanged
    assert page.locator('#cosmo-initial-preview').get_attribute('src') == source_unchanged
    put('#cosmo-mask-file',png_file('RGBA',(8,8),(255,255,255,128)));ready('#cosmo-mask-preview')
    translucent = page.locator('#cosmo-mask-preview').get_attribute('src')
    page.locator('#cosmo-mask-edit').click()
    page.wait_for_function('document.getElementById("generation-queue-status").textContent.includes("opaque")')
    assert page.locator('#cosmo-mask-editor').is_hidden()
    assert page.locator('#cosmo-mask-preview').get_attribute('src') == translucent
    page.locator('#cosmo-initial-clear').click()
    assert page.locator('#cosmo-guide-preview').is_hidden()
    assert page.locator('#cosmo-guide-file').is_disabled()
    assert not errors, errors
    browser.close()
print('BROWSER PASS: Advanced guide alpha/dimensions/strength, independent mask/guide payload, pointer paint/undo/cancel/save and immutable source; real rendered C++ page, PNG decode/preview, img2img/mask payload, task cancel, controls, repeated upload, clear, corrupt/oversized files, txt2img fallback and invalid strength; mocked API, no inference')
