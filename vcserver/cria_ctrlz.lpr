program cria_ctrlz;

{$mode objfpc}{$H+}

uses
  {$IFDEF UNIX}{$IFDEF UseCThreads}
  cthreads,
  {$ENDIF}{$ENDIF}
  Classes
  { you can add units after this };
var
  f : File of Byte;
  b : byte;
begin
  AssignFile(f,'/home/am/desenv/python/vcdoc/ctrlz.txt');
  Rewrite(f);
  b := 26;
  Write(f,b);
  CloseFile(f);
end.

