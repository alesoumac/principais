unit uymk_main;

{$mode objfpc}{$H+}

interface

uses
  Classes, SysUtils, Forms, Controls, Graphics, Dialogs, ExtCtrls, ComCtrls,
  EditBtn, StdCtrls, FileCtrl, CheckLst, Menus, uymk2, IniFiles;

const
  EDGE_FOR_RESIZE = 10;

type

  { TfrmYoloMarker }

  TfrmYoloMarker = class(TForm)
    btnInsertObject: TButton;
    btnDeleteObject: TButton;
    edtImageDir: TDirectoryEdit;
    IMG: TImage;
    Label1: TLabel;
    Label2: TLabel;
    lstObjects: TListBox;
    lstObjectNames: TListBox;
    N4: TMenuItem;
    MNU: TMainMenu;
    memObjects: TMemo;
    mnuImage: TMenuItem;
    mnuAddNewObject: TMenuItem;
    mnuDeleteObject: TMenuItem;
    mnuQuickAddObj: TMenuItem;
    mnuSelectDir: TMenuItem;
    mnuSelectObjNames: TMenuItem;
    mnuPreviousImage: TMenuItem;
    mnuNextImage: TMenuItem;
    mnuPreviousObject: TMenuItem;
    mnuNextObject: TMenuItem;
    mnuExit: TMenuItem;
    mnuObject: TMenuItem;
    N3: TMenuItem;
    N2: TMenuItem;
    N1: TMenuItem;
    ObjLbl: TLabel;
    lstFiles: TFileListBox;
    edtObjName: TFileNameEdit;
    lblYoloMarker: TLabel;
    lblSelectImgDir: TLabel;
    lblSelectObjNameFile: TLabel;
    ObjBox: TShape;
    panClient: TPanel;
    Panel1: TPanel;
    Panel2: TPanel;
    panInnerImg: TPanel;
    panTop: TPanel;
    panLeft: TPanel;
    panImage: TPanel;
    Shape1: TShape;
    Shape10: TShape;
    Shape11: TShape;
    Shape12: TShape;
    Shape13: TShape;
    Shape14: TShape;
    Shape15: TShape;
    Shape16: TShape;
    Shape17: TShape;
    Shape18: TShape;
    Shape19: TShape;
    Shape2: TShape;
    Shape20: TShape;
    Shape21: TShape;
    Shape22: TShape;
    Shape23: TShape;
    Shape24: TShape;
    Shape25: TShape;
    Shape26: TShape;
    Shape27: TShape;
    Shape28: TShape;
    Shape29: TShape;
    Shape3: TShape;
    Shape30: TShape;
    Shape31: TShape;
    Shape32: TShape;
    Shape33: TShape;
    Shape34: TShape;
    Shape35: TShape;
    Shape36: TShape;
    Shape37: TShape;
    Shape38: TShape;
    Shape39: TShape;
    Shape4: TShape;
    Shape40: TShape;
    Shape41: TShape;
    Shape42: TShape;
    Shape43: TShape;
    Shape44: TShape;
    Shape45: TShape;
    Shape46: TShape;
    Shape47: TShape;
    Shape48: TShape;
    Shape49: TShape;
    Shape5: TShape;
    Shape50: TShape;
    Shape51: TShape;
    Shape52: TShape;
    Shape53: TShape;
    Shape54: TShape;
    Shape55: TShape;
    Shape56: TShape;
    Shape57: TShape;
    Shape58: TShape;
    Shape59: TShape;
    Shape6: TShape;
    Shape60: TShape;
    Shape61: TShape;
    Shape62: TShape;
    Shape63: TShape;
    Shape64: TShape;
    Shape7: TShape;
    Shape8: TShape;
    Shape9: TShape;
    spl01: TSplitter;
    spl02: TSplitter;
    spl03: TSplitter;
    tmrVerifyImageLoading: TTimer;
    tmrChangeObjBoxColor: TTimer;
    tmrLoadFileList: TTimer;
    tmrFileChanged: TTimer;
    procedure btnDeleteObjectClick(Sender: TObject);
    procedure btnInsertObjectClick(Sender: TObject);
    procedure edtImageDirChange(Sender: TObject);
    procedure edtImageDirDblClick(Sender: TObject);
    procedure edtObjNameChange(Sender: TObject);
    procedure edtObjNameDblClick(Sender: TObject);
    procedure FormClose(Sender: TObject; var CloseAction: TCloseAction);
    procedure FormCreate(Sender: TObject);
    procedure FormKeyDown(Sender: TObject; var Key: Word; Shift: TShiftState);
    procedure lstFilesChange(Sender: TObject);
    procedure lstObjectsClick(Sender: TObject);
    procedure memObjectsChange(Sender: TObject);
    procedure memObjectsEnter(Sender: TObject);
    procedure mnuDecreaseHeightClick(Sender: TObject);
    procedure mnuDecreaseWidthClick(Sender: TObject);
    procedure mnuIncreaseHeightClick(Sender: TObject);
    procedure mnuIncreaseWidthClick(Sender: TObject);
    procedure mnuMoveObjDownClick(Sender: TObject);
    procedure mnuMoveObjLeftClick(Sender: TObject);
    procedure mnuMoveObjRightClick(Sender: TObject);
    procedure mnuMoveObjUpClick(Sender: TObject);
    procedure mnuMoveFastDownClick(Sender: TObject);
    procedure mnuMoveFastLeftClick(Sender: TObject);
    procedure mnuMoveFastRightClick(Sender: TObject);
    procedure mnuMoveFastUpClick(Sender: TObject);
    procedure mnuQuickAddObjClick(Sender: TObject);
    procedure mnuPreviousObjectClick(Sender: TObject);
    procedure mnuNextObjectClick(Sender: TObject);
    procedure mnuExitClick(Sender: TObject);
    procedure mnuNextImageClick(Sender: TObject);
    procedure mnuPreviousImageClick(Sender: TObject);
    procedure ObjBoxChangeBounds(Sender: TObject);
    procedure ObjBoxMouseDown(Sender: TObject; Button: TMouseButton;
      Shift: TShiftState; X, Y: Integer);
    procedure ObjBoxMouseLeave(Sender: TObject);
    procedure ObjBoxMouseMove(Sender: TObject; Shift: TShiftState; X, Y: Integer
      );
    procedure panImageResize(Sender: TObject);
    procedure ShapeMouseDown(Sender: TObject; Button: TMouseButton;
      Shift: TShiftState; X, Y: Integer);
    procedure tmrVerifyImageLoadingTimer(Sender: TObject);
    procedure tmrChangeObjBoxColorTimer(Sender: TObject);
    procedure tmrFileChangedTimer(Sender: TObject);
    procedure tmrLoadFileListTimer(Sender: TObject);
  private
    AppPath : string;
    FileChanged : boolean;
    ImgWidth,ImgHeight : integer;
    LastObjClass : integer;
    LastFileName : string;
    CurrentMarkFile : string;
    ObjMoveX, ObjMoveY : integer;
    ObjMoveDirection : string;
    PenColor : TColor;
    procedure ChangeBoxVisibility(Visib : Boolean);
    procedure BasicReadImage(FileName : string);
    procedure BasicReadMarkers(FileName : string);
    procedure BasicResizeImageComponent;
    procedure BasicRecalculateShapes;
    procedure BasicOpenObjNamesFile(ObjFile : string);
    procedure AutoSave;
  public

  end;

var
  frmYoloMarker: TfrmYoloMarker;

implementation

{$R *.lfm}


type
  TDictRectangle = 
    record
      objR : integer;
      x1,y1,x2,y2 : Extended;
    end;
  TDictBase =
    record
      objB : integer;
      xcenter,ycenter,width,height : Extended;
    end;

  TArrayRect = array of TDictRectangle;
  TArrayBase = array of TDictBase;

function ConvertBaseToRect(base : TArrayBase; doTrim : boolean = False) : TArrayRect;
var
  k,i,N : integer;
  NewRect : TArrayRect;
begin
  N := Length(base);
  SetLength(NewRect, N);
  i := 0;
  for k := 0 to N - 1 do
  begin
    with base[k],NewRect[i] do
    begin
      objR := objB;
      x1 := xcenter - width / 2;
      y1 := ycenter - height / 2;
      x2 := x1 + width;
      y2 := y1 + height;

      if doTrim
      then begin
        if (x1 >= 1) or (y1 >= 1) or (x2 <= 0) or (y2 <= 0)
        then
          Continue;
        if x1 < 0 then x1 := 0;
        if y1 < 0 then y1 := 0;
        if x2 > 1 then x2 := 1;
        if y2 > 1 then y2 := 1;
      end;
    end;
    Inc(i);
  end;
  SetLength(NewRect, i);
  Result := NewRect;
end;

function ConvertRectToBase(rect : TArrayRect; doTrim : boolean = False) : TArrayBase;
var
  k,i,N : integer;
  NewBase : TArrayBase;
  xx1,xx2,yy1,yy2 : Extended;
begin
  N := Length(rect);
  SetLength(NewBase, N);
  i := 0;
  for k := 0 to N - 1 do
  begin
    with rect[k],NewBase[i] do
    begin
      xx1 := x1;
      yy1 := y1;
      xx2 := x2;
      yy2 := y2;

      if doTrim
      then begin
        if (x1 >= 1) or (y1 >= 1) or (x2 <= 0) or (y2 <= 0)
        then
          Continue;
        if x1 < 0 then xx1 := 0;
        if y1 < 0 then yy1 := 0;
        if x2 > 1 then xx2 := 1;
        if y2 > 1 then yy2 := 1;
      end;

      objB    := objR;
      width   := xx2 - xx1;
      height  := yy2 - yy1;
      xcenter := (xx2 + xx1) / 2.0;
      ycenter := (yy2 + yy1) / 2.0;
    end;
    Inc(i);
  end;
  SetLength(NewBase, i);
  Result := NewBase;
end;

(*
def calculeNovaBase(baseInicial,chaveNova,baseCoordNova):
    if chaveNova not in baseInicial:
        return {chaveNova: baseCoordNova}

    rectNovo = {}

    try:
        rectInicial = convertBaseToRect(baseInicial)

        xbn,ybn,wbn,hbn = baseCoordNova
        xbi,ybi,wbi,hbi = baseInicial[chaveNova]

        x1bn,y1bn,x2bn,y2bn = xbn-wbn/2.0 , ybn-hbn/2.0 , xbn+wbn/2.0 , ybn+hbn/2.0
        x1bi,y1bi,x2bi,y2bi = rectInicial[chaveNova]


        for k in baseInicial:
            if k == chaveNova:
                rectNovo[k] = (0,0,1,1)
                continue

            try:
                x1ki,y1ki,x2ki,y2ki = rectInicial[k]
                wki = x2ki-x1ki
                hki = y2ki-y1ki
                x1 = min(x1ki,x1bi)
                y1 = min(y1ki,y1bi)
                x2 = max(x2ki,x2bi)
                y2 = max(y2ki,y2bi)
                w = x2-x1
                h = y2-y1

                wn = w * wbn / wbi
                hn = h * hbn / hbi
                x1n = x1bn - (x1bi-x1)/w * wn
                y1n = y1bn - (y1bi-y1)/h * hn
                x2n = x1n + wn
                y2n = y1n + hn

                wkn = wn * wki / w
                hkn = hn * hki / h
                x1kn = x1n + (x1ki-x1)/w * wn
                y1kn = y1n + (y1ki-y1)/h * hn
                x2kn = x1kn + wkn
                y2kn = y1kn + hkn

                rectNovo[k] = (x1kn,y1kn,x2kn,y2kn)
            except:
                pass
    except:
        pass
    baseNova = convertRectToBase(rectNovo,True)
    baseNova[chaveNova] = baseCoordNova
    return baseNova
*)

{ TfrmYoloMarker }
procedure SplitString(SplitChar : char; var S : string; var Initial : string);
var
  p : integer;
begin
  S := Trim(S);
  p := Pos(SplitChar,S);
  if p < 1 then p := Length(S) + 1;
  Initial := Trim(Copy(S,1,p-1));
  S := Trim(Copy(S,p+1,Length(S)));
end;

function MyStrToFloat(S : string; ValorDefault : Extended) : Extended;
begin
  if DefaultFormatSettings.DecimalSeparator = ','
  then
    S := StringReplace(S,'.',',',[rfReplaceAll]);
  Result := StrToFloatDef(S,ValorDefault);
end;

function MyFloatToStr(N : Extended) : string;
var
  S : string;
begin
  S := FormatFloat('0.00000000',N);
  if DefaultFormatSettings.DecimalSeparator = ','
  then
    S := StringReplace(S,',','.',[rfReplaceAll]);
  Result := S;
end;

function StrToBase(S : string;
  var num_classe : integer;
  var x : Extended;
  var y : Extended;
  var w : Extended;
  var h : Extended) : boolean;

var
  N_partes : integer;
  parte : string;
  S_partes : array [1..5] of string;

begin
  N_partes := 0;
  while (N_partes < 5) and (S <> '') do
  begin
    Inc(N_partes);
    SplitString(' ',S,parte);
    S_partes[N_partes] := parte;
  end;
  Result := False;
  if N_partes = 5
  then begin
    num_classe := StrToIntDef(S_partes[1],-1);
    x := MyStrToFloat(S_partes[2],-10000);
    y := MyStrToFloat(S_partes[3],-10000);
    w := MyStrToFloat(S_partes[4],-10000);
    h := MyStrToFloat(S_partes[5],-10000);
    Result := (num_classe >= 0)
              and (x <> -10000)
              and (y <> -10000)
              and (w <> -10000)
              and (h <> -10000)
              ;
  end
  else if N_partes = 4
  then begin
    num_classe := -1;
    x := MyStrToFloat(S_partes[1],-10000);
    y := MyStrToFloat(S_partes[2],-10000);
    w := MyStrToFloat(S_partes[3],-10000);
    h := MyStrToFloat(S_partes[4],-10000);
    Result := (x <> -10000)
              and (y <> -10000)
              and (w <> -10000)
              and (h <> -10000)
              ;
  end;
end;

function BaseToStr(X,Y,W,H : Extended) : string;
begin
  Result := MyFloatToStr(X)
         + ' ' + MyFloatToStr(Y)
         + ' ' + MyFloatToStr(W)
         + ' ' + MyFloatToStr(H)
         ;
end;

function BaseToStr(NumClasse : integer; X,Y,W,H : Extended) : string;
begin
  Result := IntToStr(NumClasse) + ' ' + BaseToStr(X,Y,W,H);
end;

procedure TfrmYoloMarker.edtImageDirChange(Sender: TObject);
var
  s : string;
begin
  s := edtImageDir.Text;
  if not DirectoryExists(s) then Exit;

  if length(s) > 1
  then
    s := ExcludeTrailingPathDelimiter(s);

  lstFiles.Directory := s;
  tmrLoadFileList.Enabled := True;
end;

procedure TfrmYoloMarker.btnInsertObjectClick(Sender: TObject);
begin
  Application.CreateForm(TForm1, Form1);
  try
    Form1.cmbObjClassName.Items.Text := lstObjectNames.Items.Text;
    if lstObjectNames.Items.Count > 0
    then begin
      if (LastObjClass < 0) or (LastObjClass >= lstObjectNames.Items.Count)
      then
        LastObjClass := 0;
      Form1.cmbObjClassName.ItemIndex := LastObjClass;
    end;
    if Form1.ShowModal = mrOK
    then begin
      LastObjClass := Form1.cmbObjClassName.ItemIndex;
      if Form1.LineToInsert <> ''
      then begin
        memObjects.Lines.Add(Form1.LineToInsert);
        BasicRecalculateShapes;
      end;
    end;
  finally
    Form1.Release;
  end;

end;

procedure TfrmYoloMarker.btnDeleteObjectClick(Sender: TObject);
var
  i,idx : integer;
  LS : string;
begin
  if lstObjects.Items.Count = 0 then Exit;
  idx := lstObjects.ItemIndex;
  if idx = -1 then Exit;
  LS := '';
  for i := 0 to memObjects.Lines.Count - 1 do
  begin
    if i = idx then Continue;
    LS := LS + memObjects.Lines[i] + #10;
  end;
  memObjects.Lines.Text := LS;
  BasicRecalculateShapes;
end;

procedure TfrmYoloMarker.edtImageDirDblClick(Sender: TObject);
begin
  edtImageDir.RunDialog;
end;

procedure TfrmYoloMarker.edtObjNameChange(Sender: TObject);
var
  s : string;
begin
  s := edtObjName.Text;
  lstObjectNames.Clear;
  if not FileExists(s) then Exit;
  BasicOpenObjNamesFile(s);
end;

procedure TfrmYoloMarker.edtObjNameDblClick(Sender: TObject);
begin
  edtObjName.RunDialog;
end;

procedure TfrmYoloMarker.FormClose(Sender: TObject;
  var CloseAction: TCloseAction);
var
  Ini : TIniFile;
begin
  Ini := TIniFile.Create(AppPath + 'ymarker.ini');
  try
    Ini.WriteString('Files','ImagesDirectory',edtImageDir.Text);
    Ini.WriteString('Files','ObjFile',edtObjName.Text);
    Ini.WriteString('Files','LastImage',LastFileName);
    Ini.WriteInteger('Files','LastObject',LastObjClass);
  finally
    Ini.Free;
  end;
end;

procedure TfrmYoloMarker.FormCreate(Sender: TObject);
var
  Ini : TIniFile;
begin
  AppPath := IncludeTrailingPathDelimiter(ExtractFilePath(ExpandFileName(
    Application.ExeName)));

  IMG.Picture.Clear;
  Ini := TIniFile.Create(AppPath + 'ymarker.ini');
  try
    LastObjClass := Ini.ReadInteger('Files','LastObject',0);
    LastFileName := Ini.ReadString('Files','LastImage','');
    edtObjName.Text := Ini.ReadString('Files','ObjFile','');
    edtImageDir.Text := Ini.ReadString('Files','ImagesDirectory',AppPath);
  finally
    Ini.Free;
  end;
end;

procedure TfrmYoloMarker.FormKeyDown(Sender: TObject; var Key: Word;
  Shift: TShiftState);
var
  S : string;
begin
  if not (ActiveControl is TListBox) then Exit;
  S := '';
  if ssShift in Shift then S := S + '^';
  if ssAlt in Shift then S := S + '@';
  if ssCtrl in Shift then S := S + '??';
  S := S+IntToStr(Ord(Key));
  //Caption :=  S + ' ' + Copy(Caption,1,32);
  if S = '??37' then mnuMoveObjLeftClick(Sender)  else
  if S = '??38' then mnuMoveObjUpClick(Sender)    else
  if S = '??39' then mnuMoveObjRightClick(Sender) else
  if S = '??40' then mnuMoveObjDownClick(Sender)  else

  if S = '^37' then mnuDecreaseWidthClick(Sender)  else
  if S = '^38' then mnuDecreaseHeightClick(Sender) else
  if S = '^39' then mnuIncreaseWidthClick(Sender)  else
  if S = '^40' then mnuIncreaseHeightClick(Sender) else

  if S = '^??37' then mnuMoveFastLeftClick(Sender)  else
  if S = '^??38' then mnuMoveFastUpClick(Sender)    else
  if S = '^??39' then mnuMoveFastRightClick(Sender) else
  if S = '^??40' then mnuMoveFastDownClick(Sender)  else
  Exit;

  Key := 0;
end;

procedure TfrmYoloMarker.lstFilesChange(Sender: TObject);
begin
  FileChanged := True;
  tmrFileChanged.Enabled := True;
end;

procedure TfrmYoloMarker.lstObjectsClick(Sender: TObject);
var
  idx : integer;
  sh : TShape;
  objName,dummy : string;
begin
  ChangeBoxVisibility(False);
  idx := lstObjects.ItemIndex;
  if idx < 0 then Exit;

  objName := lstObjects.Items[idx];
  SplitString(':',objName,dummy);

  Inc(idx);
  sh := FindComponent('Shape'+IntToStr(idx)) as TShape;

  ChangeBoxVisibility(True);

  ObjBox.SetBounds(sh.Left,sh.Top,sh.Width,sh.Height);
  ObjLbl.Caption := objName;
end;

procedure TfrmYoloMarker.memObjectsChange(Sender: TObject);
begin
  AutoSave;
  if ActiveControl = memObjects
  then begin
    BasicRecalculateShapes;
  end;
end;

procedure TfrmYoloMarker.ChangeBoxVisibility(Visib : Boolean);
begin
  ObjBox.Visible := Visib;
  ObjLbl.Visible := Visib;
end;

procedure TfrmYoloMarker.memObjectsEnter(Sender: TObject);
begin
  lstObjects.ItemIndex := -1;
  ChangeBoxVisibility(False);
end;

procedure TfrmYoloMarker.mnuDecreaseHeightClick(Sender: TObject);
begin
  if not ObjBox.Visible then Exit;

  if ObjBox.Height > 1
  then
    ObjBox.SetBounds(ObjBox.Left, ObjBox.Top, ObjBox.Width, ObjBox.Height - 1);
end;

procedure TfrmYoloMarker.mnuDecreaseWidthClick(Sender: TObject);
begin
  if not ObjBox.Visible then Exit;

  if ObjBox.Width > 1
  then
    ObjBox.SetBounds(ObjBox.Left, ObjBox.Top, ObjBox.Width - 1, ObjBox.Height);

end;

procedure TfrmYoloMarker.mnuIncreaseHeightClick(Sender: TObject);
begin
  if not ObjBox.Visible then Exit;

  if ObjBox.Top + ObjBox.Height < panInnerImg.Height
  then
    ObjBox.SetBounds(ObjBox.Left, ObjBox.Top, ObjBox.Width, ObjBox.Height + 1);

end;

procedure TfrmYoloMarker.mnuIncreaseWidthClick(Sender: TObject);
begin
  if not ObjBox.Visible then Exit;

  if ObjBox.Left + ObjBox.Width < panInnerImg.Width
  then
    ObjBox.SetBounds(ObjBox.Left, ObjBox.Top, ObjBox.Width + 1, ObjBox.Height);
end;

procedure TfrmYoloMarker.mnuMoveObjDownClick(Sender: TObject);
begin
  if not ObjBox.Visible then Exit;

  if ObjBox.Top + ObjBox.Height < panInnerImg.Height
  then
    ObjBox.SetBounds(ObjBox.Left, ObjBox.Top + 1, ObjBox.Width, ObjBox.Height);
end;

procedure TfrmYoloMarker.mnuMoveObjLeftClick(Sender: TObject);
begin
  if not ObjBox.Visible then Exit;

  if ObjBox.Left > 0
  then
    ObjBox.SetBounds(ObjBox.Left - 1, ObjBox.Top, ObjBox.Width, ObjBox.Height);
end;

procedure TfrmYoloMarker.mnuMoveObjRightClick(Sender: TObject);
begin
  if not ObjBox.Visible then Exit;

  if ObjBox.Left + ObjBox.Width < panInnerImg.Width
  then
    ObjBox.SetBounds(ObjBox.Left + 1, ObjBox.Top, ObjBox.Width, ObjBox.Height);
end;

procedure TfrmYoloMarker.mnuMoveObjUpClick(Sender: TObject);
begin
  if not ObjBox.Visible then Exit;

  if ObjBox.Top > 0
  then
    ObjBox.SetBounds(ObjBox.Left, ObjBox.Top - 1, ObjBox.Width, ObjBox.Height);
end;

procedure TfrmYoloMarker.mnuMoveFastDownClick(Sender: TObject);
begin
  if not ObjBox.Visible then Exit;

  if ObjBox.Top + ObjBox.Height <= panInnerImg.Height - 8
  then
    ObjBox.SetBounds(ObjBox.Left, ObjBox.Top + 8, ObjBox.Width, ObjBox.Height)
  else
    ObjBox.SetBounds(ObjBox.Left, panInnerImg.Height - ObjBox.Height, ObjBox.Width, ObjBox.Height);
end;

procedure TfrmYoloMarker.mnuMoveFastLeftClick(Sender: TObject);
begin
  if not ObjBox.Visible then Exit;

  if ObjBox.Left >= 8
  then
    ObjBox.SetBounds(ObjBox.Left - 8, ObjBox.Top, ObjBox.Width, ObjBox.Height)
  else
    ObjBox.SetBounds(0, ObjBox.Top, ObjBox.Width, ObjBox.Height);
end;

procedure TfrmYoloMarker.mnuMoveFastRightClick(Sender: TObject);
begin
  if not ObjBox.Visible then Exit;

  if ObjBox.Left + ObjBox.Width <= panInnerImg.Width - 8
  then
    ObjBox.SetBounds(ObjBox.Left + 8, ObjBox.Top, ObjBox.Width, ObjBox.Height)
  else
    ObjBox.SetBounds(panInnerImg.Width - ObjBox.Width, ObjBox.Top, ObjBox.Width, ObjBox.Height);
end;

procedure TfrmYoloMarker.mnuMoveFastUpClick(Sender: TObject);
begin
  if not ObjBox.Visible then Exit;

  if ObjBox.Top >= 8
  then
    ObjBox.SetBounds(ObjBox.Left, ObjBox.Top - 8, ObjBox.Width, ObjBox.Height)
  else
    ObjBox.SetBounds(ObjBox.Left, 0, ObjBox.Width, ObjBox.Height)
end;

procedure TfrmYoloMarker.mnuQuickAddObjClick(Sender: TObject);
var
  L : string;
begin
  if (LastObjClass < 0) or (LastObjClass >= lstObjectNames.Count)
  then
    LastObjClass := 0;
  L := IntToStr(LastObjClass) + ' 0.5 0.5 0.2 0.2';
  memObjects.Lines.Add(L);
  BasicRecalculateShapes;
end;

procedure TfrmYoloMarker.mnuPreviousObjectClick(Sender: TObject);
var
  idx : integer;
begin
  if lstObjects.Items.Count = 0 then Exit;
  idx := lstObjects.ItemIndex;
  if idx > -1 then Dec(idx);
  if idx = -1 then idx := lstObjects.Items.Count - 1;
  lstObjects.ItemIndex := idx;
  lstObjects.Click;
end;

procedure TfrmYoloMarker.mnuNextObjectClick(Sender: TObject);
var
  idx : integer;
begin
  if lstObjects.Items.Count = 0 then Exit;
  idx := lstObjects.ItemIndex + 1;
  if idx >= lstObjects.Items.Count then idx := 0;
  lstObjects.ItemIndex := idx;
  lstObjects.Click;
end;

procedure TfrmYoloMarker.mnuExitClick(Sender: TObject);
begin
  Close;
end;

procedure TfrmYoloMarker.mnuNextImageClick(Sender: TObject);
var
  idx : integer;
begin
  if lstFiles.Items.Count = 0 then Exit;
  idx := lstFiles.ItemIndex + 1;
  if idx >= lstFiles.Items.Count then idx := 0;
  lstFiles.ItemIndex := idx;
  lstFiles.Click;
end;

procedure TfrmYoloMarker.mnuPreviousImageClick(Sender: TObject);
var
  idx : integer;
begin
  if lstFiles.Items.Count = 0 then Exit;
  idx := lstFiles.ItemIndex;
  if idx > -1 then Dec(idx);
  if idx = -1 then idx := lstFiles.Items.Count - 1;
  lstFiles.ItemIndex := idx;
  lstFiles.Click;
end;

procedure TfrmYoloMarker.ObjBoxChangeBounds(Sender: TObject);
var
  idx : integer;
  sh : TShape;
  L,dummy : string;
  px,py,pw,ph : double;
begin
  ObjLbl.Left := ObjBox.Left + 4;

  if ObjBox.Height <= ObjLbl.Height + 16
  then begin
    if ObjBox.Top + ObjBox.Height + ObjLbl.Height + 4 > panInnerImg.Height
    then begin
      if ObjBox.Top - ObjLbl.Height - 4 < 0
      then
        ObjLbl.Top := ObjBox.Top + ObjBox.Height - 4 - ObjLbl.Height
      else
        ObjLbl.Top := ObjBox.Top - ObjLbl.Height - 4;
    end
    else begin
      ObjLbl.Top := ObjBox.Top + ObjBox.Height + 4;
    end;
  end
  else
    ObjLbl.Top := ObjBox.Top + (ObjBox.Height div 2) - (ObjLbl.Height div 2);

  idx := lstObjects.ItemIndex;
  if idx = -1 then Exit;
  sh := FindComponent('Shape'+IntToStr(idx+1)) as TShape;
  with ObjBox do
  begin
    sh.SetBounds(Left,Top,Width,Height);
    px := (Left + (Width / 2)) / panInnerImg.Width;
    py := (Top + (Height / 2)) / panInnerImg.Height;
    pw := Width / panInnerImg.Width;
    ph := Height / panInnerImg.Height;
  end;
  L := memObjects.Lines[idx];
  SplitString(' ',L,dummy);
  L := dummy + ' ' + BaseToStr(px,py,pw,ph);
  memObjects.Lines[idx] := L;
end;

procedure TfrmYoloMarker.ObjBoxMouseDown(Sender: TObject; Button: TMouseButton;
  Shift: TShiftState; X, Y: Integer);
begin
  if ssLeft in Shift
  then begin
    ObjMoveX := X;
    ObjMoveY := Y;

    ObjMoveDirection := '';
    if X > ObjBox.Width - EDGE_FOR_RESIZE - 1
    then ObjMoveDirection := 'E'
    else if X < EDGE_FOR_RESIZE then ObjMoveDirection := 'W';

    if Y > ObjBox.Height - EDGE_FOR_RESIZE - 1
    then ObjMoveDirection := 'S' + ObjMoveDirection
    else if Y < EDGE_FOR_RESIZE then ObjMoveDirection := 'N' + ObjMoveDirection;
  end;
end;

procedure TfrmYoloMarker.ObjBoxMouseLeave(Sender: TObject);
begin
  Screen.Cursor := crDefault;
end;

procedure TfrmYoloMarker.ObjBoxMouseMove(Sender: TObject; Shift: TShiftState;
  X, Y: Integer);
var
  Resizing_ : boolean;
  DifX, DifY : integer;

  function TryBounds(NewLeft,NewTop,NewWidth,NewHeight : integer) : integer;
  begin
    Result := 3;
    if (NewLeft < 0)
    or (NewWidth < 2)
    or (NewLeft + NewWidth > panInnerImg.Width)
    then
      Result := Result - 2;

    if (NewTop < 0)
    or (NewHeight < 2)
    or (NewTop + NewHeight > panInnerImg.Height)
    then
      Result := Result - 1;
    if odd(Result)
    then
       with ObjBox do SetBounds(Left, NewTop, Width, NewHeight);
    if Result > 1
    then
      with ObjBox do SetBounds(NewLeft, Top, NewWidth, Height);
  end;

begin
  Resizing_ := ssLeft in Shift;
  if Resizing_
  then begin
    DifX := X - ObjMoveX;
    DifY := Y - ObjMoveY;
  end
  else begin
    ObjMoveDirection := '';
    if X > ObjBox.Width - EDGE_FOR_RESIZE - 1
    then ObjMoveDirection := 'E'
    else if X < EDGE_FOR_RESIZE then ObjMoveDirection := 'W';

    if Y > ObjBox.Height - EDGE_FOR_RESIZE - 1
    then ObjMoveDirection := 'S' + ObjMoveDirection
    else if Y < EDGE_FOR_RESIZE then ObjMoveDirection := 'N' + ObjMoveDirection;
  end;


  if ObjMoveDirection = ''
  then begin
    if Resizing_
    then
      TryBounds(ObjBox.Left + DifX, ObjBox.Top + DifY, ObjBox.Width, ObjBox.Height);
    ObjBox.Cursor:= crSize;
  end
  else if ObjMoveDirection = 'N'
  then begin
    if Resizing_
    then
      TryBounds(ObjBox.Left, ObjBox.Top + DifY, ObjBox.Width, ObjBox.Height - DifY);
    ObjBox.Cursor := crSizeN;
  end
  else if ObjMoveDirection = 'W'
  then begin
    if Resizing_
    then
      TryBounds(ObjBox.Left + DifX, ObjBox.Top, ObjBox.Width - DifX, ObjBox.Height);
    ObjBox.Cursor := crSizeW;
  end
  else if ObjMoveDirection = 'S'
  then begin
    if Resizing_
    then
      if TryBounds(ObjBox.Left, ObjBox.Top, ObjBox.Width, ObjBox.Height + DifY) and 1 = 1
      then
        ObjMoveY := objMoveY + DifY;
    ObjBox.Cursor := crSizeS;
  end
  else if ObjMoveDirection = 'E'
  then begin
    if Resizing_
    then
      if TryBounds(ObjBox.Left, ObjBox.Top, ObjBox.Width + DifX, ObjBox.Height) and 2 = 2
      then
        ObjMoveX := objMoveX + DifX;
    ObjBox.Cursor := crSizeE;
  end
  else if ObjMoveDirection = 'NW'
  then begin
    if Resizing_
    then
      TryBounds(ObjBox.Left + DifX, ObjBox.Top + DifY, ObjBox.Width - DifX, ObjBox.Height - DifY);
    ObjBox.Cursor := crSizeNW;
  end
  else if ObjMoveDirection = 'NE'
  then begin
    if Resizing_
    then
      if TryBounds(ObjBox.Left, ObjBox.Top + DifY, ObjBox.Width + DifX, ObjBox.Height - DifY) and 2 = 2
      then
        ObjMoveX := objMoveX + DifX;
    ObjBox.Cursor := crSizeNE;
  end
  else if ObjMoveDirection = 'SW'
  then begin
    if Resizing_
    then
      if TryBounds(ObjBox.Left + DifX, ObjBox.Top, ObjBox.Width - DifX, ObjBox.Height + DifY) and 1 = 1
      then
        ObjMoveY := objMoveY + DifY;
    ObjBox.Cursor := crSizeSW;
  end
  else if ObjMoveDirection = 'SE'
  then begin
    if Resizing_
    then
      case TryBounds(ObjBox.Left, ObjBox.Top, ObjBox.Width + DifX, ObjBox.Height + DifY) of
        1: ObjMoveY := objMoveY + DifY;
        2: ObjMoveX := objMoveX + DifX;
        3:
          begin
            ObjMoveX := objMoveX + DifX;
            ObjMoveY := objMoveY + DifY;
          end;
      end;
    ObjBox.Cursor := crSizeSE;
  end
end;

procedure TfrmYoloMarker.panImageResize(Sender: TObject);
begin
  BasicResizeImageComponent;
  BasicRecalculateShapes;
end;

procedure TfrmYoloMarker.ShapeMouseDown(Sender: TObject; Button: TMouseButton;
  Shift: TShiftState; X, Y: Integer);
var
  idx : integer;
  objName : string;
begin
  if ssLeft in Shift
  then begin
    if (Sender is TShape)
    then
      objName := TShape(Sender).Name
    else
      Exit;
    if copy(objName,1,5) <> 'Shape' then Exit;
    idx := StrToInt(Copy(objName,6,Length(objName)-5));
    lstObjects.ItemIndex := idx - 1;
    lstObjects.Click;
  end;
end;

procedure TfrmYoloMarker.tmrVerifyImageLoadingTimer(Sender: TObject);
var
  FileNM : string;
begin
  if ((lstFiles.Items.Count > 0) and (lstFiles.ItemIndex <> -1)) <> (panInnerImg.Visible and (IMG.Picture.Width <> 0))
  then begin
    panInnerImg.Visible := (lstFiles.Items.Count > 0) and (lstFiles.ItemIndex <> -1);
    memObjects.Enabled := panInnerImg.Visible;
    if panInnerImg.Visible
    then begin
      LastFileName := lstFiles.Items[lstFiles.ItemIndex];
      FileNM := IncludeTrailingPathDelimiter(edtImageDir.Text) + LastFileName;
      BasicReadImage(FileNM);
      BasicResizeImageComponent;
      BasicReadMarkers(FileNM);
      BasicRecalculateShapes;
    end
    else begin
      LastFileName := '';
      CurrentMarkFile := '';
    end;
  end;
end;

procedure TfrmYoloMarker.tmrChangeObjBoxColorTimer(Sender: TObject);
begin
  if PenColor = clFuchsia then PenColor := clPurple else PenColor := clFuchsia;
  ObjBox.Pen.Color := PenColor;
  ObjLbl.Color := PenColor;
end;

procedure TfrmYoloMarker.tmrFileChangedTimer(Sender: TObject);
var
  idx : integer;
begin
  if not FileChanged then Exit;

  FileChanged := False;
  tmrFileChanged.Enabled := False;

  idx := lstFiles.ItemIndex;

  if idx = -1
  then LastFileName := ''
  else LastFileName := lstFiles.Items[idx];

  BasicReadImage(lstFiles.FileName);
  BasicResizeImageComponent;
  BasicReadMarkers(lstFiles.FileName);
  BasicRecalculateShapes;
end;

procedure TfrmYoloMarker.tmrLoadFileListTimer(Sender: TObject);
var
  idx : integer;
begin
  tmrLoadFileList.Enabled := False;

  if LastFileName = ''
  then begin
    lstFiles.ItemIndex := -1;
  end
  else begin
    idx := lstFiles.Items.IndexOf(LastFileName);
    lstFiles.ItemIndex := idx;
  end;
  lstFilesChange(nil);
end;

procedure TfrmYoloMarker.BasicReadImage(FileName: string);
var
  OldCursor : TCursor;
begin
  IMG.Picture.Clear;
  IMG.Stretch := False;
  ImgWidth := 0;
  ImgHeight := 0;

  if FileName = '' then Exit;

  OldCursor := Screen.Cursor;
  try
    Screen.Cursor := crHourGlass;

    IMG.Picture.LoadFromFile(FileName);
    ImgWidth  := IMG.Picture.Width;
    ImgHeight := IMG.Picture.Height;
    IMG.Stretch := True;

  finally
    Screen.Cursor := OldCursor;
  end;
end;

procedure TfrmYoloMarker.BasicReadMarkers(FileName: string);
var
  MarkFileName : string;
  i : integer;
  X,Y,W,H : Extended;
  NumClasse : integer;
  L : string;
  TxtRecognized : boolean;
  SaveMemoChange : TNotifyEvent;
begin
  SaveMemoChange := memObjects.OnChange;
  try
    memObjects.OnChange := nil;
    memObjects.Clear;
    if FileName = '' then Exit;
    MarkFileName := ChangeFileExt(FileName,'.txt');
    CurrentMarkFile := MarkFileName;
    if not FileExists(MarkFileName) then Exit;

    memObjects.Lines.LoadFromFile(MarkFileName);
    TxtRecognized := True;
    for i := 0 to memObjects.Lines.Count - 1 do
    begin
      L := memObjects.Lines[i];
      if L = '' then continue;
      TxtRecognized := StrToBase(L,NumClasse,X,Y,W,H);
      if not TxtRecognized then Break;
    end;

    if not TxtRecognized
    then begin
      ShowMessage('File "' + MarkFileName
        + '" exists, but it''s not a Label Mark File.'#13#10
        + 'Renaming as backup.');
      RenameFile(MarkFileName,MarkFileName+'.backup');
      memObjects.Clear;
      Exit;
    end;

  finally
    memObjects.OnChange := SaveMemoChange;
  end;
end;

procedure TfrmYoloMarker.BasicRecalculateShapes;
var
  i,idx,classNumber,NewLeft,NewTop,NewWidth,NewHeight : integer;
  L,classNumberStr,objClassName,numberStr : string;
  px,py,pw,ph : Extended;
  sh : TShape;
  linhaOk : boolean;
begin
  ChangeBoxVisibility(False);
  lstObjects.Clear;
  for i := 1 to 64 do
  begin
    sh := FindComponent('Shape'+IntToStr(i)) as TShape;
    sh.Visible := False;
  end;
  classNumberStr := '';
  numberStr := '';
  for i := 0 to memObjects.Lines.Count - 1 do
  begin
    L := memObjects.Lines[i];
    linhaOk := StrToBase(L,classNumber,px,py,pw,ph);
    if (classNumber < 0) or (classNumber > lstObjectNames.Count-1)
    then
      objClassName := 'Unknown class "' + classNumberStr + '"'
    else
      objClassName := IntToStr(i+1) + ': ' + lstObjectNames.Items[classNumber];

    if (px < 0) or (py < 0) or (pw < 0) or (ph < 0) then Continue;

    NewWidth  := Trunc(pw * panInnerImg.Width);
    NewHeight := Trunc(ph * panInnerImg.Height);
    NewLeft   := Trunc(px * panInnerImg.Width) - (NewWidth div 2);
    NewTop    := Trunc(py * panInnerImg.Height) - (NewHeight div 2);

    lstObjects.Items.Add(objClassName);
    idx := lstObjects.Items.Count;
    sh := FindComponent('Shape'+IntToStr(idx)) as TShape;
    sh.SetBounds(NewLeft,NewTop,NewWidth,NewHeight);
    sh.Visible := True;
    sh.Pen.Width := 1;
    sh.Pen.Color := clRed;
    sh.Pen.Mode  := pmCopy;
    sh.Brush.Style := bsClear;
    sh.Cursor := crHandPoint;
    sh.OnMouseDown := Shape1.OnMouseDown;
  end;
end;

{
procedure TfrmYoloMarker.BasicRecalculateZOrder;
var
  i,j,idx : integer;
  ShapeArea : array [1..64] of Double;
  ZSort : array [1..64] of integer;
  sh : TShape;
begin
  for i := 0 to lstObjects.Count - 1 do
  begin
    idx := i+1;
    sh := FindComponent('Shape'+IntToStr(idx));
    ZSort[idx] := idx;
    ShapeArea[idx] := sh.Width * sh.Height;
  end;

  for i := 1 to lstObjects.Count do
  begin
    for j :=
      sh.
  end;
end;
}
procedure TfrmYoloMarker.BasicResizeImageComponent;
var
  NewLeft, NewTop, NewWidth, NewHeight : integer;
  MinScale, ScaleWidth, ScaleHeight : Double;
  OldCursor : TCursor;
begin
  if (ImgWidth <= 1) or (ImgHeight <= 1) or (panImage.Width <= 10) or (panImage.Height <= 10)
  then begin
    Exit;
  end;

  OldCursor := Screen.Cursor;

  try
    ScaleWidth  := panImage.Width / ImgWidth;
    ScaleHeight := panImage.Height / ImgHeight;
    if ScaleWidth < ScaleHeight then MinScale := ScaleWidth else MinScale := ScaleHeight;
    NewWidth  := Trunc(MinScale * ImgWidth);
    NewHeight := Trunc(MinScale * ImgHeight);
    NewLeft   := (panImage.Width - NewWidth) div 2;
    NewTop    := (panImage.Height - NewHeight) div 2;

    panInnerImg.SetBounds(NewLeft,NewTop,NewWidth,NewHeight);
    IMG.SetBounds(0,0,NewWidth,NewHeight);

  finally
    Screen.Cursor := OldCursor;
  end;
end;

procedure TfrmYoloMarker.BasicOpenObjNamesFile(ObjFile: string);
var
  ObjF : TextFile;
  aName : string;
begin
  try
    AssignFile(ObjF,ObjFile);
    Reset(ObjF);
    lstObjectNames.Clear;
    while not Eof(ObjF) do
    begin
      Readln(ObjF,aName);
      lstObjectNames.Items.Add(aName);
    end;
    CloseFile(ObjF);
    panInnerImg.Visible := False;
  except
    ShowMessage('Unable to read Object Names file');
  end;
end;

procedure TfrmYoloMarker.AutoSave;
begin
  if (CurrentMarkFile = '') or not panInnerImg.Visible then Exit;
  memObjects.Lines.SaveToFile(CurrentMarkFile);
end;

end.

