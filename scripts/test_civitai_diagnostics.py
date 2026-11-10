"""Regression coverage for response diagnostics, image scope, and file provenance."""
import importlib
import json
import tempfile
import unittest
from pathlib import Path
from unittest import mock

import requests
from PIL import Image, PngImagePlugin
from test_online_model_browser import service, api, PACKAGE_NAME


class CivitaiDiagnosticsTests(unittest.TestCase):
    def setUp(self):
        service._REQUEST_CACHE.clear()
        service._BRIDGED_IMAGE_CACHE.clear()

    def test_logs_received_bytes_without_query_or_auth_secrets(self):
        response = requests.Response()
        response.status_code = 200
        response._content = b'{"items":[],"metadata":{}}'
        response.headers.update({'Content-Length': '17', 'Content-Encoding': 'gzip', 'Content-Type': 'application/json'})
        with mock.patch.object(service.requests, 'request', return_value=response), mock.patch.object(service, '_log') as log:
            service._request_with_retries('GET', 'https://civitai.com/api/v1/images?token=private',
                                          params={'query': 'private prompt'}, api_key='private-token')
        output = '\n'.join(str(call.args[0]) for call in log.call_args_list)
        self.assertIn('received_bytes=26', output)
        self.assertIn('content_length=17', output)
        self.assertIn('content_encoding=gzip', output)
        self.assertIn('status=200', output)
        self.assertIn('items=0', output)
        self.assertNotIn('private', output)

    def test_logs_invalid_json_size_before_failure(self):
        response = requests.Response()
        response.status_code = 200
        response._content = b'<html>upstream error</html>'
        with mock.patch.object(service.requests, 'request', return_value=response), mock.patch.object(service, '_log') as log:
            with self.assertRaises(ValueError):
                service._request_with_retries('GET', 'https://civitai.com/api/v1/images', max_retries=0)
        self.assertTrue(any('received_bytes=27' in str(call) for call in log.call_args_list))

    def test_selected_model_allows_query_and_preserves_image_cursor(self):
        with mock.patch.object(service, '_request_with_retries', return_value={'items': [], 'metadata': {}}) as request:
            service._images_civitai(query='urn:air:sdxl:checkpoint:civitai:123@456', api_key=None,
                                   model_id=123, model_version_id=456, cursor='image-next')
        params = request.call_args.kwargs['params']
        self.assertEqual(params['modelId'], 123)
        self.assertEqual(params['modelVersionId'], 456)
        self.assertEqual(params['cursor'], 'image-next')
        self.assertNotIn('query', params)
        self.assertNotIn('page', params)

    def test_blank_model_browse_preserves_cursor(self):
        with mock.patch.object(service, '_request_with_retries', return_value={'items': [], 'metadata': {}}) as request:
            service._search_civitai(query='', api_key=None, page=2, cursor='model-next')
        self.assertEqual(request.call_args.kwargs['params'].get('cursor'), 'model-next')
        self.assertNotIn('page', request.call_args.kwargs['params'])

    def test_image_tag_ids_are_forwarded_and_isolate_cached_results(self):
        with mock.patch.object(service, '_request_with_retries', return_value={'items': [], 'metadata': {}}) as request:
            service._images_civitai(query='', api_key=None, model_id=123, tag_ids=[3, 1, 2, 1])
            first = request.call_args.kwargs['params']
            service._images_civitai(query='', api_key=None, model_id=123, tag_ids=[4])
            self.assertEqual(request.call_count, 2)
            self.assertEqual(request.call_args.kwargs['params']['tags'], '4')
        self.assertEqual(first['tags'], '1,2,3')
        self.assertEqual(first['withMeta'], 'true')
        self.assertEqual(first['withTags'], 'true')

    def test_image_api_rejects_tag_names_with_400(self):
        import asyncio
        class Request:
            headers = {}
            query_params = {'modelId':'123','tags':'portrait'}
        with mock.patch.object(service, '_images_civitai', return_value={'ok':True,'items':[]}), self.assertRaises(api.HTTPException) as error:
            asyncio.run(api.civitai_images(Request()))
        self.assertEqual(error.exception.status_code, 400)

    def test_enum_response_includes_model_file_types(self):
        with mock.patch.object(service, '_request_with_retries', return_value={'ModelFileType':['Model','VAE']}):
            self.assertEqual(service._enums_civitai()['modelFileTypes'], ['Model','VAE'])

    def test_random_image_sort_is_forwarded_not_silently_replaced(self):
        with mock.patch.object(service, '_request_with_retries', return_value={'items': [], 'metadata': {}}) as request:
            service._images_civitai(query='', api_key=None, sort='Random', tag_ids=[1,2,3])
        self.assertEqual(request.call_args.kwargs['params'].get('sort'), 'Random')

    def test_exact_file_metadata_is_not_civitai_metadata(self):
        metadata = importlib.import_module(f'{PACKAGE_NAME}.image_metadata')
        with tempfile.TemporaryDirectory() as directory:
            file = Path(directory) / 'source.png'
            text = PngImagePlugin.PngInfo()
            text.add_text('parameters', 'embedded prompt, not the website prompt\nSteps: 8, Seed: 42')
            text.add_text('workflow', '{"nodes":[{"id":1}]}')
            text.add_itxt('XML:com.adobe.xmp', '<x:xmpmeta>original metadata</x:xmpmeta>')
            Image.new('RGB', (16, 24)).save(file, pnginfo=text)
            before = file.read_bytes()
            result = metadata.inspect_image_file(file)
            self.assertEqual(file.read_bytes(), before)
        self.assertEqual(result['sizeBytes'], len(before))
        self.assertEqual(result['format'], 'PNG')
        self.assertEqual(result['width'], 16)
        self.assertEqual(result['height'], 24)
        self.assertIn('embedded prompt', result['embedded']['info']['parameters'])
        self.assertIn('workflow', result['embedded']['info'])
        self.assertIn('XML:com.adobe.xmp', result['embedded']['info'])
        self.assertEqual(len(result['sha256']), 64)

    def test_exif_user_comment_is_decoded_without_losing_bytes(self):
        metadata = importlib.import_module(f'{PACKAGE_NAME}.image_metadata')
        with tempfile.TemporaryDirectory() as directory:
            file = Path(directory) / 'source.jpg'
            exif = Image.Exif()
            exif[37510] = b'ASCII\x00\x00\x00Original prompt'
            Image.new('RGB', (8, 8)).save(file, exif=exif)
            result = metadata.inspect_image_file(file)
        comment = result['embedded']['exif']['UserComment']
        self.assertEqual(comment['text'], 'Original prompt')
        self.assertIn('base64', comment)

    def test_external_redirect_is_rejected_before_request(self):
        metadata = importlib.import_module(f'{PACKAGE_NAME}.image_metadata')
        response = mock.MagicMock(status_code=302, headers={'Location': 'http://127.0.0.1/private'})
        response.__enter__.return_value = response
        with tempfile.TemporaryDirectory() as directory, mock.patch.object(service.requests, 'get', return_value=response) as get:
            with self.assertRaisesRegex(ValueError, 'official Civitai HTTPS'):
                metadata.download_image('https://image.civitai.com/test.png', Path(directory) / 'image.part')
        self.assertEqual(get.call_count, 1)
        self.assertFalse(get.call_args.kwargs['allow_redirects'])

    def test_image_report_preserves_extra_civitai_fields(self):
        meta = {'prompt': 'website', 'Model': 'urn:air:sdxl:checkpoint:civitai:827184@2167369',
                'vaes': ['urn:air:sdxl:vae:civitai:1684460@1906483'],
                'civitaiResources': [{'type':'lora', 'modelVersionId':2831497, 'weight':1.5}]}
        raw = {'id': 1, 'meta': meta, 'resources': [{'modelVersionId': 42}], 'generationProcess': 'txt2img',
               'modelVersionIds':[2167369,2831497], 'tags':[{'id':1,'name':'example'}]}
        result = service._trim_image_item(raw)
        self.assertEqual(result['resources'], raw['resources'])
        self.assertEqual(result['generationProcess'], 'txt2img')
        self.assertEqual(result['modelVersionIds'], raw['modelVersionIds'])
        self.assertEqual(result['tags'], raw['tags'])
        self.assertEqual(result['meta'], meta)
        self.assertEqual(result['meta']['civitaiResources'][0]['weight'], 1.5)

    def test_import_preserves_file_and_website_reports_separately(self):
        metadata = importlib.import_module(f'{PACKAGE_NAME}.image_metadata')
        image = {'id': 123, 'url': 'https://image.civitai.com/test.png', 'meta': {'prompt': 'website only'}}
        class Request:
            headers = {}
            async def json(self):
                return {'image': image}
        def download(url, destination, key):
            text = PngImagePlugin.PngInfo()
            text.add_text('parameters', 'file only')
            Image.new('RGB', (8, 8)).save(destination, format='PNG', pnginfo=text)
            return metadata.inspect_image_file(destination)
        with tempfile.TemporaryDirectory() as directory:
            with mock.patch.object(api.gallery, 'configured_directory', return_value=Path(directory), create=True), \
                 mock.patch.object(api, 'download_image', side_effect=download), mock.patch.object(api, '_resolve_key', return_value=None):
                import asyncio
                result = asyncio.run(api.civitai_import_image(Request()))
            saved = Path(directory) / result['relativePath']
            sidecar = json.loads(saved.with_suffix('.png.civitai.json').read_text())
            self.assertEqual(sidecar['image']['meta']['prompt'], 'website only')
            self.assertEqual(sidecar['downloadedFile']['embedded']['info']['parameters'], 'file only')
            self.assertEqual(result['fileMetadata']['sha256'], metadata.inspect_image_file(saved)['sha256'])
            self.assertEqual(result['metadata']['prompt'], 'website only')

    def test_metadata_free_image_does_not_invent_prompt(self):
        metadata = importlib.import_module(f'{PACKAGE_NAME}.image_metadata')
        with tempfile.TemporaryDirectory() as directory:
            file = Path(directory) / 'plain.png'
            Image.new('RGB', (8, 8)).save(file)
            result = metadata.inspect_image_file(file)
        self.assertEqual(result['embedded'], {'info': {}, 'exif': {}})

    def test_declared_oversized_image_is_not_written(self):
        metadata = importlib.import_module(f'{PACKAGE_NAME}.image_metadata')
        response = mock.MagicMock(status_code=200, headers={'Content-Length': str(metadata.MAX_IMAGE_BYTES + 1)})
        response.__enter__.return_value = response
        with tempfile.TemporaryDirectory() as directory, mock.patch.object(service.requests, 'get', return_value=response):
            file = Path(directory) / 'image.part'
            with self.assertRaisesRegex(ValueError, '100 MB'):
                metadata.download_image('https://image.civitai.com/test.png', file)
            self.assertFalse(file.exists())


if __name__ == '__main__':
    unittest.main()
