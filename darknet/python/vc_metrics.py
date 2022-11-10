# Reestruturação da aplicacao
# -*- coding: utf-8 -*-

import nvgpu
import prometheus_client
from prometheus_client import Counter, Summary
from prometheus_client.core import CollectorRegistry

METRICS = {}

def inicializa_metricas():
     # parte de código para as metricas
    global METRICS
    METRICS = {}
    METRICS['counter'] = Counter('vcdoc_count_requests','Número total de solicitações processadas')
    METRICS['detect_yolo_latency'] = Summary('vcdoc_detect_yolo_latency_seconds','Tempo total de detecção do YOLO')
    METRICS['detect_ocr_latency'] = Summary('vcdoc_detect_ocr_latency_seconds','Tempo total de detecção do OCR')
    METRICS['detect_total_latency'] = Summary('vcdoc_detect_total_latency_seconds','Tempo total de detecção YOLO/OCR')
    METRICS['gpu_count'] = Summary('vcdoc_gpu_count_units','Número de GPU\'s no cluster')
    METRICS['gpu_memory_used_percent'] = Summary('vcdoc_gpu_memory_used_percent_value','Percentual de uso de memória da GPU')
    METRICS['gpu_memory_free'] = Summary('vcdoc_gpu_memory_free_value','Memória livre da GPU')
    METRICS['gpu_memory_used'] = Summary('vcdoc_gpu_memory_used_value','Memória usada da GPU')
    METRICS['gpu_memory_total'] = Summary('vcdoc_gpu_memory_total_value','Memória total da GPU')

def gera_metricas_gpu():
    global METRICS
    # ----- obtendo métricas da GPU
    gpu_info = nvgpu.gpu_info()
    qtd_gpus = len(gpu_info)
    METRICS['gpu_count'].observe(qtd_gpus)
    if qtd_gpus > 0:
        METRICS['gpu_memory_used_percent'].observe(gpu_info[0]['mem_used_percent'])
        METRICS['gpu_memory_used'].observe(gpu_info[0]['mem_used'])
        METRICS['gpu_memory_total'].observe(gpu_info[0]['mem_total'])
        METRICS['gpu_memory_free'].observe(gpu_info[0]['mem_total'] - gpu_info[0]['mem_used'])

def includeRequestTimesMetrics(detect_yolo_time,detect_ocr_time = None):
    global METRICS
    ocr_time = detect_ocr_time if detect_ocr_time is not None else 0
    METRICS['counter'].inc()
    METRICS['detect_yolo_latency'].observe(detect_yolo_time)
    METRICS['detect_ocr_latency'].observe(ocr_time)
    METRICS['detect_total_latency'].observe(ocr_time + detect_yolo_time)


def snapshot():
    global METRICS
    gera_metricas_gpu()
    res = []
    for k,v in METRICS.items():
        res.append(prometheus_client.generate_latest(v))
    return ''.join([metrica.decode() for metrica in res])

