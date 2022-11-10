#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from datetime import datetime

class ClockTimer():
    def __init__(self):
        self.CLOCKS = {}

    def resetClock(self,clockName):
        self.CLOCKS[clockName] = {'total' : 0.0, 'initial' : None, 'final' : None, 'lapses_count' : 0}

    def startClock(self,clockName):
        if clockName not in self.CLOCKS:
            self.resetClock(clockName)
        if self.CLOCKS[clockName]['initial'] is not None:
            return
        self.CLOCKS[clockName]['initial'] = datetime.now()
        self.CLOCKS[clockName]['final']   = None

    def endClock(self,clockName):
        if clockName not in self.CLOCKS or self.CLOCKS[clockName]['initial'] is None:
            return
        self.CLOCKS[clockName]['final'] = datetime.now()
        lapso = self.CLOCKS[clockName]['final'] - self.CLOCKS[clockName]['initial']
        self.CLOCKS[clockName]['initial'] = None
        self.CLOCKS[clockName]['final'] = None
        self.CLOCKS[clockName]['total'] += lapso.total_seconds()
        self.CLOCKS[clockName]['lapses_count'] += 1

    def getElapsedTime(self, clockName):
        if clockName not in self.CLOCKS:
            return 0.0, 0
        elif self.CLOCKS[clockName]['initial'] is None:
            return self.CLOCKS[clockName]['total'], self.CLOCKS[clockName]['lapses_count']
        lap_time = datetime.now()
        lapso = lap_time - self.CLOCKS[clockName]['initial']
        return self.CLOCKS[clockName]['total'] + lapso.total_seconds(), self.CLOCKS[clockName]['lapses_count'] + 1

    def getTotal(self,clockName):
        return self.getElapsedTime(clockName)[0]

    def getLapsesCount(self, clockName):
        return self.getElapsedTime(clockName)[1]
    
    @property
    def totals(self):
        t = {}
        for clockName in self.CLOCKS:
            t[clockName] = self.CLOCKS[clockName]['total']
        return t

    @totals.setter
    def totals(self, value):
        raise AttributeError('ClockTimer.totals property is read-only')

    def __str__(self):
        r = ""
        for clock_name in self.CLOCKS:
            total, num_lapsos = self.getElapsedTime(clock_name)
            if num_lapsos <= 1 and total < 0.001:
                continue
            if r != "": r += " "
            lps = f'({num_lapsos} laps)' if num_lapsos > 1 else ''
            r += f'[{clock_name}{lps}: {total:5.3f}s]'
        return r

    def __repr__(self):
        return f'ClockTimer({self.__str__()})'
