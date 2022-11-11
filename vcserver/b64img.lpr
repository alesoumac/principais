program b64img;

{$mode objfpc}{$H+}

uses
  {$IFDEF UNIX}{$IFDEF UseCThreads}
  cthreads,
  {$ENDIF}{$ENDIF}
  Classes, SysUtils, CustApp
  { you can add units after this };

type

  { b64img }

  b64img = class(TCustomApplication)
  protected
    procedure DoRun; override;
  public
    constructor Create(TheOwner: TComponent); override;
    destructor Destroy; override;
    procedure WriteHelp; virtual;
  end;

{ b64img }

procedure b64img.DoRun;
var
  ErrorMsg: String;
begin
  // quick check parameters
  ErrorMsg:=CheckOptions('h', 'help');
  if ErrorMsg<>'' then begin
    ShowException(Exception.Create(ErrorMsg));
    Terminate;
    Exit;
  end;

  // parse parameters
  if HasOption('h', 'help') then begin
    WriteHelp;
    Terminate;
    Exit;
  end;

  { add your program here }


  // stop program loop
  Terminate;
end;

constructor b64img.Create(TheOwner: TComponent);
begin
  inherited Create(TheOwner);
  StopOnException:=True;
end;

destructor b64img.Destroy;
begin
  inherited Destroy;
end;

procedure b64img.WriteHelp;
begin
  { add your help code here }
  writeln('Usage: ', ExeName, ' [-h]');
end;

var
  Application: b64img;
begin
  Application:=b64img.Create(nil);
  Application.Title:='b64img';
  Application.Run;
  Application.Free;
end.

