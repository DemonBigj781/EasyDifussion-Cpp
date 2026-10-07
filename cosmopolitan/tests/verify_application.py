#!/usr/bin/env python3
"""Same-process portable application acceptance: shell, HTTP and shared state.

Default: logical shell cwd, shared config/index, tiny trained text inference,
and EOF-driven graceful shutdown. --model adds two bounded real SD1.5 jobs:
shell generation observed over HTTP, then HTTP generation canceled from shell.
No host command shell, fake model, download or compilation is used by this test.
"""
import argparse
import hashlib
import json
import math
import os
from pathlib import Path
import platform
import queue
import signal
import shutil
import socket
import subprocess
import tempfile
import threading
import time
import urllib.error
import urllib.request
import uuid

from verify_runtime import APPLICATION, LOADER, EXPECTED_GREEDY_IDS, runtime_environment, sha256, verify_package
from verify_inference import inspect_png, read_model_pin

PREFIX = '/v1/sdapi/v1/'


def require(value, message):
    if not value:
        raise RuntimeError(message)


def quote(text):
    return '"' + str(text).replace('\\', '\\\\').replace('"', '\\"') + '"'


def remaining(deadline, maximum):
    value = min(maximum, deadline - time.monotonic())
    if value <= 0:
        raise TimeoutError('Overall application acceptance deadline exceeded')
    return value


def free_port():
    with socket.socket() as stream:
        stream.bind(('127.0.0.1', 0))
        return stream.getsockname()[1]


class Http:
    def __init__(self, port, deadline, records):
        self.base, self.deadline, self.records = f'http://127.0.0.1:{port}', deadline, records

    def call(self, path, payload=None, *, expected=200, timeout=10):
        body = None if payload is None else json.dumps(payload).encode()
        request = urllib.request.Request(self.base + path, body, {'Content-Type':'application/json'})
        start = time.monotonic()
        observation = {'path':path, 'method':'GET' if body is None else 'POST', 'request':payload}
        try:
            try:
                response = urllib.request.urlopen(request, timeout=remaining(self.deadline, timeout))
            except urllib.error.HTTPError as error:
                response = error
            with response:
                status, raw = response.code, response.read(64*1024*1024 + 1)
            require(len(raw) <= 64*1024*1024, 'HTTP response exceeds bounded size')
            try:
                value = json.loads(raw)
            except (json.JSONDecodeError, UnicodeDecodeError):
                value = raw.decode(errors='replace')
            observation.update(status=status, response_bytes=len(raw), response_sha256=hashlib.sha256(raw).hexdigest())
            if expected is not None:
                require(status == expected, f'{path}: HTTP {status}, expected {expected}: {str(value)[:400]}')
            return (status, value) if expected is None else value
        except Exception as error:
            observation['error'] = str(error)
            raise
        finally:
            observation['elapsed_seconds'] = round(time.monotonic()-start, 6)
            self.records.append(observation)


class Pending:
    """A daemon avoids unbounded interpreter shutdown if a test transport breaks."""
    def __init__(self, function):
        self.done = threading.Event()
        self.value = self.error = None
        def target():
            try:
                self.value = function()
            except BaseException as error:
                self.error = error
            finally:
                self.done.set()
        self.thread = threading.Thread(target=target, daemon=True)
        self.thread.start()

    def result(self, deadline):
        require(self.done.wait(remaining(deadline, 1800)), 'Pending application request exceeded deadline')
        if self.error:
            raise self.error
        return self.value


class Shell:
    def __init__(self, command, work, logs, env, deadline, records):
        self.deadline, self.records, self.lines = deadline, records, queue.Queue()
        self.errors = queue.Queue()
        self.lock = threading.Lock()
        self.process = subprocess.Popen(command, cwd=work, env=env, stdin=subprocess.PIPE,
                                        stdout=subprocess.PIPE, stderr=subprocess.PIPE,
                                        start_new_session=os.name != 'nt')
        self.files = []
        self.threads = []
        for name, pipe in [('stdout',self.process.stdout),('stderr',self.process.stderr)]:
            target = (logs / ('application.'+name+'.log')).open('xb')
            self.files.append(target)
            def read(stream=pipe, output=target, capture=name=='stdout'):
                try:
                    for line in iter(stream.readline, b''):
                        output.write(line); output.flush()
                        text=line.decode(errors='replace').rstrip('\r\n')
                        if capture: self.lines.put(text)
                        elif text.startswith('{'):
                            try:
                                value=json.loads(text)
                                if isinstance(value,dict) and isinstance(value.get('message'),str): self.errors.put(value)
                            except json.JSONDecodeError: pass
                finally:
                    stream.close()
                    if capture: self.lines.put(None)
            thread=threading.Thread(target=read,daemon=True);thread.start();self.threads.append(thread)

    def command(self, text, *, failure=False, error_message=None, timeout=15):
        with self.lock:
            while True:
                try: self.errors.get_nowait()
                except queue.Empty: break
            token = uuid.uuid4().hex
            begin, end, handled = '__BEGIN_'+token, '__END_'+token, '__HANDLED_'+token
            script = 'echo '+begin+'; '+text+(' || echo '+handled if failure else '')+'; echo '+end+'\n'
            observation = {'command':text,'expected_failure':failure}
            self.records.append(observation)
            require(self.process.poll() is None, 'Application exited before shell command')
            self.process.stdin.write(script.encode());self.process.stdin.flush()
            stop = min(self.deadline, time.monotonic()+timeout)
            collected, started = [], False
            while True:
                try: line = self.lines.get(timeout=remaining(stop, timeout))
                except queue.Empty: raise TimeoutError('Shell reply exceeded deadline: '+text[:160])
                require(line is not None, 'Application stdout ended before command reply: '+text[:160])
                if line == begin: started = True;continue
                if not started: continue
                if line == end: break
                collected.append(line)
            require((handled in collected) == failure, 'Expected shell failure was not observed')
            documents=[]
            for line in collected:
                if line.startswith(('{','[')):
                    try: documents.append(json.loads(line))
                    except json.JSONDecodeError: pass  # Non-JSON application logs remain in raw stdout.
            if failure:
                try: error=self.errors.get(timeout=remaining(stop,2))
                except queue.Empty: raise RuntimeError('Expected JSON shell error was not observed')
                observation['error_document']=error
                if error_message is not None:
                    require(error['message']==error_message,'Shell failure did not report the expected reason: '+str(error))
            observation.update(reply_documents=documents, completed=True)
            return documents

    def one(self, text, **kwargs):
        documents=self.command(text,**kwargs)
        require(len(documents)==1, 'Expected one JSON reply for '+text[:120])
        return documents[0]

    def close(self, *, graceful):
        result = {'exit_status_before_cleanup':self.process.poll(), 'forced_termination':False}
        try:
            if self.process.stdin and not self.process.stdin.closed: self.process.stdin.close()
            if self.process.poll() is None:
                try: self.process.wait(timeout=remaining(self.deadline, 20))
                except (subprocess.TimeoutExpired,TimeoutError):
                    result['forced_termination']=True
                    if os.name=='nt': self.process.kill()
                    else: os.killpg(self.process.pid,signal.SIGKILL)
                    self.process.wait(timeout=5)
            result['exit_status']=self.process.returncode
        finally:
            for thread in self.threads: thread.join(timeout=3)
            for output in self.files:
                if not any(t.is_alive() for t in self.threads): output.close()
        if graceful:
            require(not result['forced_termination'] and result['exit_status']==0,
                    'EOF did not gracefully stop the application/HTTP service: '+str(result))
        return result


def check_text(value):
    require(isinstance(value,dict) and value.get('token_ids')==EXPECTED_GREEDY_IDS,
            'Shared text service differs from independent sixteen-token reference')
    require(value.get('completion_tokens')==16 and value.get('cancelled') is False,
            'Text service did not finish sixteen uncanceled tokens')
    require(value.get('prompt_tokens')==5 and value.get('backend')=='cpu' and value.get('provider')=='builtin' and value.get('device')=='CPU' and value.get('software') is True,
            'Text service did not report the requested CPU fixture execution')
    require(isinstance(value.get('text'),str) and value['text'], 'Text service returned no generated text')
    return value


def task_progress(http, task):
    code, value = http.call('/v1/internal/progress', {'id_task':task,'live_preview':False}, expected=None, timeout=2)
    if code==404: return None
    require(code==200 and isinstance(value,dict) and type(value.get('completed')) is bool, 'Invalid shared task progress')
    for field in ('current_step','total_steps'):
        require(type(value.get(field)) in (int,float) and math.isfinite(value[field]) and value[field]>=0,
                'Invalid task step field')
    return value


def wait_active(http, shell, pending, task, deadline, *, sampling=False):
    while time.monotonic()<deadline:
        require(shell.process.poll() is None, 'Application exited during shared job')
        require(not pending.done.is_set(), 'Shared job completed before required active observation')
        state=task_progress(http,task)
        if state and not state['completed'] and (not sampling or state['current_step']>0): return state
        time.sleep(0.05)
    raise TimeoutError('Shared task did not enter the requested phase')


def image_checks(args, shell, http, config, work, output, report, deadline):
    model=args.model.resolve()
    pin, metadata=read_model_pin(args)
    require(model.stat().st_size==pin['bytes'] and sha256(model)==pin['sha256'], 'External model differs from pinned checkpoint')
    report['model']={'path':str(model),'pin':metadata,'sha256_before':pin['sha256']}
    listing=shell.one('models refresh')
    require(listing==http.call(PREFIX+'checkpoints') and any(x.get('name')==model.name for x in listing['models']),
            'Shell and HTTP do not share the actual checkpoint index')
    recipe={'options':{'sd_model_checkpoint':model.name},'inference':{'image':{'prompt':'a red apple on a wooden table',
            'negative_prompt':'','width':256,'height':256,'steps':2,'cfg_scale':7.0,'seed':42,'sampler_name':'euler_a','scheduler':'discrete'}}}
    saved=shell.one('defaults save '+quote(json.dumps(recipe)))
    require(saved==http.call(PREFIX+'config'), 'Saved shell recipe differs from HTTP config')
    require(saved['saved']['options']['sd_model_checkpoint']==model.name and saved['effective']['inference']['image']==recipe['inference']['image'],
            'The selected model and complete image recipe were not saved/applied together')
    baseline=sha256(config)
    first='shell-image-'+uuid.uuid4().hex
    # Deliberately omit prompt/model/dimensions: this must consume shared saved defaults.
    pending=Pending(lambda:shell.one('infer image --output shell.png --task-id '+first,timeout=remaining(deadline,1800)))
    active=wait_active(http,shell,pending,first,deadline)
    state=http.call('/v1/application/status')
    require(state['busy'] is True and state['active_task_id']==first, 'HTTP did not observe the shell-owned job')
    http.call(PREFIX+'txt2img',{'force_task_id':'blocked-'+first,'override_settings':{'sd_model_checkpoint':'__missing__.gguf'}},expected=409)
    http.call('/v1/text/completions',{'force_task_id':'blocked-text-'+first,'prompt':'Once upon a time','max_tokens':16,'threads':1},expected=409)
    http.call(PREFIX+'options',{'CLIP_stop_at_last_layers':2},expected=409)
    http.call('/app_config',{'server':{'port':8189}},expected=409)
    polls=[]
    for _ in range(4):
        require(not pending.done.is_set(), 'Shared busy window ended before progress checks')
        start=time.monotonic();p=task_progress(http,first);polls.append(round(time.monotonic()-start,6))
        require(p and not p['completed'], 'Progress did not remain responsive during shell generation')
    result=pending.result(deadline)
    require(result['task_id']==first and result.get('output_bytes',0)>0, 'Shell image returned a different task/output')
    used=json.loads(json.loads(result['info'])['infotexts'])
    require(all(used.get(key)==recipe['inference']['image'][key] for key in ('prompt','negative_prompt','width','height','steps','seed','cfg_scale')),
            'Actual shell image generation did not inherit the saved recipe')
    png=work/'outputs with spaces'/'shell.png'
    decoded=inspect_png(png,256,256)
    require(result['output_bytes']==decoded['bytes'], 'Shell output metadata differs from real PNG')
    (output/'shell-image.png').write_bytes(png.read_bytes())
    require(task_progress(http,first)['completed'] is True and sha256(config)==baseline, 'Image did not complete cleanly or altered saved defaults')
    report['shell_image']={'task_id':first,'initial_progress':active,'result':result,'png':decoded,'http_busy_rejections':4,'cross_engine_text_rejected':True,'active_progress_seconds':polls}

    second='http-image-'+uuid.uuid4().hex
    body={'force_task_id':second,'steps':8,'prompt':'a red apple on a wooden table','width':256,'height':256,
          'seed':42,'batch_size':1,'sampler_name':'euler_a','scheduler':'discrete'}
    pending=Pending(lambda:http.call(PREFIX+'txt2img',body,expected=None,timeout=remaining(deadline,1800)))
    wait_active(http,shell,pending,second,deadline)
    status=shell.one('status')
    require(status['busy'] is True and status['active_task_id']==second, 'Shell did not observe HTTP-owned job')
    shell.command('infer image --output blocked.png --model __missing__.gguf',failure=True,error_message='A native generation request is active')
    shell.command('options set CLIP_stop_at_last_layers 3',failure=True,error_message='A native generation request is active')
    require(not (work/'outputs with spaces'/'blocked.png').exists() and sha256(config)==baseline, 'Busy shell request changed output/config')
    sampling=wait_active(http,shell,pending,second,deadline,sampling=True)
    shell_progress=shell.one('progress '+second)
    require(not shell_progress['completed'] and 0<shell_progress['current_step']<8, 'Shell progress missed active sampling')
    shell.one('cancel '+second)
    code,canceled=pending.result(deadline)
    final=task_progress(http,second)
    require(final['completed'] is True and final.get('interrupted') is True and 0<final['current_step']<8,
            'Shell cancellation was not early/completed in the shared HTTP task table')
    require((code==200 and not canceled.get('images')) or (code==500 and isinstance(canceled,dict) and canceled.get('message')),
            'Canceled HTTP generation returned unexpected success images/status')
    require(shell.one('status')['busy'] is False and http.call('/v1/application/status')['busy'] is False,
            'Shared request admission gate was not released')
    require(sha256(config)==baseline, 'HTTP request overrides/cancel changed persisted configuration')
    report['http_image_shell_cancel']={'task_id':second,'sampling_progress':sampling,'shell_progress':shell_progress,
                                      'final_progress':final,'http_status':code,'response':canceled,'gate_released':True}
    report['model']['sha256_after']=sha256(model)
    require(report['model']['sha256_after']==pin['sha256'], 'Model changed during shared application test')
    report['external_model_tested']=True


def verify(args):
    artifact,output=args.artifact_dir.resolve(),args.output_dir.resolve()
    require(not output.exists() or (output.is_dir() and not any(output.iterdir())), 'Use a fresh or empty output directory')
    output.mkdir(parents=True,exist_ok=True)
    report={'schema':1,'status':'failed','platform':platform.platform(),'scope':'One actual application shared by C shell and HTTP',
            'external_model_tested':False,'quality_assessed':False,'physical_hardware_tested':False,'shell_commands':[],'http_requests':[],
            'verifier':{'file':Path(__file__).name,'sha256':sha256(Path(__file__))}}
    start=time.monotonic();deadline=start+args.timeout;shell=None;work=None
    app=artifact/APPLICATION
    try:
        report['application_sha256_before']=verify_package(artifact,args.expected_sha256)
        report['build_source']=json.loads((artifact/'BUILD.json').read_text()).get('source')
        work=Path(tempfile.mkdtemp(prefix='cosmo-application-'))
        (work/'outputs with spaces').mkdir();(work/'models').mkdir()
        config=work/'settings with spaces.json';port=free_port()
        models=args.model.resolve().parent if args.model else work/'models'
        command=([str(app)] if os.name=='nt' else [str(artifact/LOADER),str(app)])
        command+=['shell','--serve','--config',str(config),'--backend','cpu','--provider','embedded','--port',str(port),'--ckpt-dir',str(models)]
        env=runtime_environment(work);env['LP_NUM_THREADS']='2'
        report.update(command=command,software_driver_threads=2)
        shell=Shell(command,work,output,env,deadline,report['shell_commands'])
        http=Http(port,deadline,report['http_requests'])
        ready=min(deadline,time.monotonic()+20)
        while time.monotonic()<ready:
            require(shell.process.poll() is None, 'Application exited during shell/HTTP startup')
            try: initial=http.call('/v1/application/status',timeout=0.5);break
            except (OSError,TimeoutError):time.sleep(0.05)
        else:raise TimeoutError('Shared HTTP service did not become ready')
        require(initial.get('shared_services') is True and initial.get('application_id'), 'Missing shared service identity')
        require(shell.one('status')==initial and initial['busy'] is False, 'Shell and HTTP do not share an idle application')
        report['application_id']=initial['application_id']
        document=shell.one('config show');require(document==http.call('/get/app_config')==http.call(PREFIX+'config'),'Config frontends disagree')
        require(document['saved']==json.loads(config.read_text()),'Shared config differs from disk')
        shell.one('options set CLIP_stop_at_last_layers 2')
        options=shell.one('options show');require(options==http.call(PREFIX+'options') and options['CLIP_stop_at_last_layers']==2,'Shell options not visible through HTTP')
        http.call(PREFIX+'options',{'CLIP_stop_at_last_layers':3})
        require(shell.one('options show')['CLIP_stop_at_last_layers']==3,'HTTP options not visible through shell')
        stable=sha256(config)
        shell.command('config set compute.backend unsupported',failure=True)
        require(sha256(config)==stable,'Rejected shell config update changed disk')
        old_effective=shell.one('config show')['effective']
        changed=shell.one('config set compute.provider native')
        require(changed['saved']['compute']['provider']=='native' and changed['effective']==old_effective and changed['restart_required'] is True,
                'Saved provider change altered running provider or missed restart requirement')
        require(changed==http.call(PREFIX+'config'),'HTTP missed shell provider update')
        listing=shell.one('models');require(listing==http.call(PREFIX+'checkpoints'),'Model indices differ')
        shell.command('cd "outputs with spaces"; pwd')
        require(shell.one('config show')==changed and shell.one('models')==listing,'Shell cd changed service/config roots')
        recipe={'inference':{'image':{'prompt':'literal --device value','width':256,'height':256,'steps':2}}}
        defaults=shell.one('defaults save '+quote(json.dumps(recipe)))
        require(defaults==http.call(PREFIX+'config') and shell.one('defaults show')==defaults['effective']['inference'], 'Shared saved recipe differs')
        saved=sha256(config);shell.command('defaults save '+quote('{"inference":{"image":{"steps":0}}}'),failure=True)
        require(sha256(config)==saved,'Rejected recipe changed disk')
        report['shared_configuration']={'initial':document,'after_mutations':defaults,'invalid_updates_unchanged':True,'provider_restart_required':True,'shell_cd_preserved_service_roots':True}
        first='shell-text-'+uuid.uuid4().hex
        text1=check_text(shell.one('infer text --prompt "Once upon a time" --tokens 16 --threads 1 --task-id '+first,timeout=120))
        require(text1['task_id']==first and shell.one('progress '+first)['completed'] is True,'Shell text task not shared/completed')
        second='http-text-'+uuid.uuid4().hex
        text2=check_text(http.call('/v1/text/completions',{'prompt':'Once upon a time','max_tokens':16,'threads':1,'force_task_id':second},timeout=120))
        require(text2['task_id']==second and shell.one('progress '+second)['completed'] is True,'HTTP text task not visible in shell')
        require(shell.one('status')['application_id']==initial['application_id'],'Application context was replaced between commands')
        require(sha256(config)==saved,'Text inference persisted request overrides')
        report['trained_text']={'shell':text1,'http':text2,'same_application_id':True,'independent_reference_match':True}
        if args.model:
            image_checks(args,shell,http,config,work,output,report,deadline)
            final_text_id='post-diffusion-text-'+uuid.uuid4().hex
            final_text=check_text(shell.one('infer text --prompt \"Once upon a time\" --tokens 16 --threads 1 --task-id '+final_text_id,timeout=120))
            require(final_text['task_id']==final_text_id and shell.one('progress '+final_text_id)['completed'] is True,
                    'Post-diffusion text task did not complete in the shared task table')
            report['trained_text']['after_diffusion']=final_text
            report['trained_text']['registry_reuse_after_diffusion']=True
        require(shell.one('status')['application_id']==initial['application_id'],'Shared app identity changed')
        report['final_configuration']=json.loads(config.read_text())
        (output/'configuration.json').write_bytes(config.read_bytes())
        report['shutdown']=shell.close(graceful=True);shell=None
        with socket.socket() as check:
            check.settimeout(1)
            require(check.connect_ex(('127.0.0.1',port))!=0,'HTTP listener survived shell EOF')
        report['shutdown']['http_listener_closed']=True
        require(not list(work.rglob('*.tmp.*')),'Application left temporary files')
        report['status']='passed'
    except Exception as error:
        report['error']=str(error)
    finally:
        if shell is not None:
            try:
                report['failure_cleanup']=shell.close(graceful=False)
            except Exception as error:
                report['failure_cleanup']={'error':str(error),'exit_status':shell.process.poll()}
        # Windows cannot remove a directory that is still a running child's
        # cwd. Reap the child first, and never replace the original gate error
        # with a secondary cleanup error. No automatic context-manager cleanup
        # runs while the failure is unwinding.
        if work is not None:
            if shell is not None and shell.process.poll() is None:
                report['workspace_cleanup']={'removed':False,'path':str(work),'reason':'Application is still running'}
                report['status']='failed'
                report.setdefault('error','Application could not be stopped before workspace cleanup')
            else:
                try:
                    shutil.rmtree(work)
                    report['workspace_cleanup']={'removed':True}
                except Exception as error:
                    report['workspace_cleanup']={'removed':False,'path':str(work),'error':str(error)}
                    report['status']='failed'
                    report.setdefault('error','Workspace cleanup failed: '+str(error))
        if report.get('model'):
            try:
                report['model']['sha256_after']=sha256(args.model.resolve())
                if report['model']['sha256_after']!=report['model']['sha256_before']:
                    report.update(status='failed',error='Model identity changed during shared application test')
            except Exception as error:
                report.update(status='failed',model_identity_error=str(error))
        if app.exists():
            report['application_sha256_after']=sha256(app)
            if report.get('application_sha256_before')!=report['application_sha256_after']:
                report.update(status='failed',error='Application identity changed or was not verified')
        report['elapsed_seconds']=round(time.monotonic()-start,6)
        report['evidence']={p.name:{'bytes':p.stat().st_size,'sha256':sha256(p)} for p in sorted(output.iterdir()) if p.is_file()}
        (output/'report.json').write_text(json.dumps(report,indent=2)+'\n')
    return report


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--artifact-dir',type=Path,required=True)
    parser.add_argument('--output-dir',type=Path,required=True)
    parser.add_argument('--expected-sha256')
    parser.add_argument('--model',type=Path)
    parser.add_argument('--model-pin',type=Path)
    parser.add_argument('--timeout',type=float,default=1800)
    args=parser.parse_args()
    if not 0<args.timeout<=3600:parser.error('--timeout must be in (0,3600]')
    result=verify(args)
    print(json.dumps({'status':result['status'],'external_model_tested':result['external_model_tested'],'error':result.get('error')}))
    return 0 if result['status']=='passed' else 1


if __name__=='__main__':
    raise SystemExit(main())
