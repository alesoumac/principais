#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from types import TracebackType
import vc_utils

class InvalidBase64ImageError(Exception): pass       # HTTP_404
class ImageQualityCheckerError(Exception): pass      # HTTP_412
class RequestDoesntHaveImageError(Exception): pass   # HTTP_400

class InvalidKeyException(Exception): pass           # HTTP_401
class AuthenticationError(Exception): pass           # HTTP_401

class PreprocessingKindTypeError(Exception): pass    # HTTP_400

class NeededToRunYoloDetectionError(Exception): pass # HTTP_500

def convertExceptionToHtmlResponse(exception):
    vc_utils.printLog(str(exception.with_traceback(None)))
    msg = str(exception)
    retCode = 513
    if   isinstance(exception, InvalidBase64ImageError):       retCode = 404
    elif isinstance(exception, ImageQualityCheckerError):      retCode = 412
    elif isinstance(exception, RequestDoesntHaveImageError):   retCode = 400
    elif isinstance(exception, InvalidKeyException):           retCode = 401
    elif isinstance(exception, AuthenticationError):           retCode = 401
    elif isinstance(exception, PreprocessingKindTypeError):    retCode = 400
    elif isinstance(exception, NeededToRunYoloDetectionError): retCode = 500

    if retCode == 513:
        msg = f'Uncommon error: {msg}'
    return vc_utils.responseError(retCode,msg)