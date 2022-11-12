import datetime
import json
import time

import cv2
import numpy as np

YOLO_NET = None
YOLO_LAYER_NAMES = None
YOLO_LABELS = None

def Load_DNN(objNames, yoloCfg, yoloWeights):
    global YOLO_NET
    global YOLO_LAYER_NAMES
    global YOLO_LABELS
    if YOLO_NET is None:
        YOLO_NET = cv2.dnn.readNet(yoloCfg, yoloWeights)
        YOLO_LAYER_NAMES = YOLO_NET.getLayerNames()
        try:
            YOLO_LAYER_NAMES = [YOLO_LAYER_NAMES[i[0] - 1] for i in YOLO_NET.getUnconnectedOutLayers()]
        except:
            YOLO_LAYER_NAMES = [YOLO_LAYER_NAMES[i - 1] for i in YOLO_NET.getUnconnectedOutLayers()]
        YOLO_LABELS = open(objNames).read().strip().split('\n')

# Ler imagem

def convert_json_types(obj):
    if isinstance(obj, np.int64):
        return int(obj)
    elif isinstance(obj, np.integer):
        return int(obj)
    elif isinstance(obj, np.floating):
        return float(obj)
    elif isinstance(obj, np.ndarray):
        return obj.tolist()
    elif isinstance(obj, datetime.datetime):
        return obj.__str__()
    else:
        return obj

def key_name(x):
    idx = x["class_index"]
    nome = x["obj_name"].lower()
    if nome.startswith("cnh"):
        return idx
    elif nome.endswith("_cnh"):
        return idx + 100
    elif nome.startswith("rg_"):
        return idx + 200
    elif nome.endswith("_rg"):
        return idx + 300
    elif nome.startswith("rg2_"):
        return idx + 400
    else:
        return idx + 500

def yoloDetectDNN(img_np_bgr, threshold = 0.5):
    global YOLO_NET
    global YOLO_LAYER_NAMES
    global YOLO_LABELS

    resultado = ""
    try:
        if YOLO_NET is None:
            raise Exception("Rede neural não foi carregada")

        imagem = img_np_bgr
        (H, W) = imagem.shape[:2]
        inicio = time.time()

        blob = cv2.dnn.blobFromImage(imagem, 1 / 255.0, (416, 416), swapRB = True, crop = False)
        YOLO_NET.setInput(blob)
        layer_outputs = YOLO_NET.forward(YOLO_LAYER_NAMES)

        termino = time.time()
        predict_time_secs = termino - inicio
        #print('YOLO levou {:.2f} segundos'.format(termino - inicio))

        threshold_NMS = 0.3
        caixas = []
        confiancas = []
        IDclasses = []

        for output in layer_outputs:
            for detection in output:
                scores = detection[5:]
                classeID = np.argmax(scores)
                confianca = scores[classeID]
                if confianca > threshold:
                    #print('scores: ' + str(scores))
                    #print('classe mais provável: ' + str(classeID))
                    #print('confiança: ' + str(confianca))

                    caixa = detection[0:4] * np.array([W, H, W, H])
                    (centerX, centerY, width, height) = caixa.astype('int')

                    x = int(centerX - (width / 2))
                    y = int(centerY - (height / 2))

                    caixas.append([x, y, int(width), int(height)])
                    confiancas.append(float(confianca))
                    IDclasses.append(classeID)

        objs = cv2.dnn.NMSBoxes(caixas, confiancas, threshold, threshold_NMS)

        detects = []
        if len(objs) > 0:
            for i in objs.flatten():
                (x, y, w, h) = (caixas[i][0], caixas[i][1], caixas[i][2], caixas[i][3])
                bbox = {"x_min":x, "y_min":y, "width":w, "height":h}
                objdic = {
                    "class_index" : IDclasses[i],
                    "obj_name" : YOLO_LABELS[IDclasses[i]],
                    "bounding_box" : bbox,
                    "score" : confiancas[i]
                    }
                detects += [objdic]
        detects.sort(key=lambda x:key_name(x))
        resultado = { "status": "success", "resultlist": detects, "predict_time_secs": predict_time_secs }
        
    except Exception as err:
        resultado = { "status": "exception", "msg": str(err) }
    resjson = json.loads( json.dumps(resultado, indent=2, default=convert_json_types) )
    return resjson
