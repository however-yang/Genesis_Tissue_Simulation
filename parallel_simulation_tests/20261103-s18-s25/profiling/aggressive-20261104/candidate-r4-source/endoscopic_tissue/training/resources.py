"""Read-only GPU sampling, including compute AND graphics contexts."""
import os
import subprocess
import time
import xml.etree.ElementTree as ET
import psutil
import pwd


def number(text):
    return float(text.split()[0])


def gpu_sample(uuid):
    q=subprocess.run(['nvidia-smi','-i',uuid,'-q','-x'],check=True,capture_output=True,text=True,timeout=15)
    gpu=ET.fromstring(q.stdout).find('gpu')
    if gpu is None or gpu.findtext('uuid')!=uuid:raise RuntimeError('GPU resource identity mismatch')
    s=dict(uuid=uuid,total_mib=number(gpu.findtext('fb_memory_usage/total')),
           used_mib=number(gpu.findtext('fb_memory_usage/used')),
           gpu_percent=number(gpu.findtext('utilization/gpu_util')),
           memory_percent=number(gpu.findtext('utilization/memory_util')),
           temp_c=number(gpu.findtext('temperature/gpu_temp')),power_w=None,processes=[],background_display_processes=[])
    for name in ('gpu_power_readings','power_readings'):
        value=gpu.findtext(name+'/power_draw')
        if value and value!='N/A':s['power_w']=number(value);break
    for process in gpu.findall('processes/process_info'):
        memory=process.findtext('used_memory')
        row=dict(pid=int(process.findtext('pid')),type=process.findtext('type'),
                 memory_mib=None if memory=='N/A' else number(memory))
        # This server's persistent GDM Xorg registers 4 MiB on every GPU.
        # Preserve it in the evidence, without treating it as a sampling job.
        # All other graphics jobs remain subject to the foreign-job guard.
        background=False
        if row['type']=='G' and row['memory_mib'] is not None and row['memory_mib']<=16:
            try:
                proc=psutil.Process(row['pid'])
                background=(proc.uids().real==pwd.getpwnam('gdm').pw_uid
                            and proc.cmdline()[0]=='/usr/lib/xorg/Xorg')
            except (psutil.Error,KeyError,IndexError):pass
        s['background_display_processes' if background else 'processes'].append(row)
    s.update(time=time.time(),host_available_gib=psutil.virtual_memory().available/2**30,
             host_cpu_percent=psutil.cpu_percent(),loadavg=list(os.getloadavg()))
    return s
