unit uapi1_main;

{$mode objfpc}{$H+}

interface

uses
  Classes, SysUtils, FileUtil, Forms, Controls, Graphics, IniFiles, Dialogs,
  ExtCtrls, StdCtrls, EditBtn, FileCtrl, fpopenssl, fphttpclient, base64,
  fpjson, jsonparser, Grids, Menus, Buttons, ComCtrls, process;

var
  Ignore_Autentikus : boolean = True;
  Use_ValAutentikus : boolean = True;

type
  TAPYStyle = (asDetect, asDetectOCR);

type

  { TfrmMain }

  TfrmMain = class(TForm)
    Bevel1: TBevel;
    btnDetect: TButton;
    btnDetectOCR: TButton;
    chkResizing: TCheckBox;
    chkRotation: TCheckBox;
    edtMaxLarg: TEdit;
    edtImagem: TEdit;
    edtDir: TDirectoryEdit;
    grpDoc: TRadioGroup;
    Label1: TLabel;
    lblBaseAddress: TLabel;
    lblPercentMatch: TLabel;
    lblStatus: TLabel;
    lblMaxLarg: TLabel;
    lstArquivos: TFileListBox;
    IMG: TImage;
    lblArquivos: TLabel;
    lblImagem: TLabel;
    lblJSON: TLabel;
    panAmbientes: TPanel;
    sep01: TMenuItem;
    mnuSaveAndBack: TMenuItem;
    mnuSair: TMenuItem;
    mnuAPI: TMenuItem;
    mnuDetect: TMenuItem;
    mnuDetectOCR: TMenuItem;
    sep02: TMenuItem;
    mnuSaveAndNext: TMenuItem;
    mnuAbrirDiretorio: TMenuItem;
    mnuArquivo: TMenuItem;
    MNU: TMainMenu;
    memJSON: TMemo;
    panBotoes: TPanel;
    panAcoes: TPanel;
    panJSON: TPanel;
    panInfos: TPanel;
    panImagem: TPanel;
    splArquivos: TSplitter;
    splLeft: TSplitter;
    splTop: TSplitter;
    grdValores: TStringGrid;
    tabAmbientes: TTabControl;
    tmrBearerValido: TTimer;
    procedure btnDetectClick(Sender: TObject);
    procedure btnDetectOCRClick(Sender: TObject);
    procedure chkResizingChange(Sender: TObject);
    procedure edtDirChange(Sender: TObject);
    procedure edtMaxLargEditingDone(Sender: TObject);
    procedure FormActivate(Sender: TObject);
    procedure FormClose(Sender: TObject; var CloseAction: TCloseAction);
    procedure FormCreate(Sender: TObject);
    procedure FormResize(Sender: TObject);
    procedure grdValoresHeaderSized(Sender: TObject; IsColumn: Boolean;
      Index: Integer);
    procedure grdValoresResize(Sender: TObject);
    procedure grpDocClick(Sender: TObject);
    procedure IMGPictureChanged(Sender: TObject);
    procedure lstArquivosChange(Sender: TObject);
    procedure grdValoresEnter(Sender: TObject);
    procedure grdValoresExit(Sender: TObject);
    procedure grdValoresKeyPress(Sender: TObject; var Key: char);
    procedure mnuAbrirDiretorioClick(Sender: TObject);
    procedure mnuSairClick(Sender: TObject);
    procedure mnuSaveAndBackClick(Sender: TObject);
    procedure mnuSaveAndNextClick(Sender: TObject);
    procedure SaveJSON;
    procedure splArquivosChangeBounds(Sender: TObject);
    procedure splLeftChangeBounds(Sender: TObject);
    procedure splTopChangeBounds(Sender: TObject);
    procedure tabAmbientesChange(Sender: TObject);
    procedure tmrBearerValidoTimer(Sender: TObject);
    procedure SelecionaBaseAddress;
    //procedure HttpClientGetSocketHandler(Sender: TObject;
    //  const UseSSL: Boolean; out AHandler: TSocketHandler);

  private
    BearerValido : boolean;
    BearerAtual : string;
    OldJSON,OldFilename : string;
    Valores : TJSONData;
    Redimensionando : Boolean;
    Comecou : Boolean;
    PathProg : string;
    ImagesDir : string;
    ApiEnvironment,ApiAddress,ApiDetectOCR,ApiDetect : string;
    AddressP,AddressD,AddressH,AddressL : string;

    FileListHeight, ImagePanelWidth, JSONPanelHeight,
    ColGridValoresWidth0, ColGridValoresWidth1,
    ColGridValoresWidth2, ColGridValoresWidth3
      : Real;

    procedure AddLine(Dados: TJSONData; Key : string);
    procedure LimpaTudo(ImagemTambem : Boolean = False);
    procedure MakeAutentikusHeader(var HTTPObj : TFPHTTPClient);
    procedure RunApi(ApiURI : string; Estilo : TAPYStyle);
    function CreateJsonFromValores : string;
    procedure ClearGrid;
    function FindGridRow(Chave : string) : integer;
  public
  end;

var
  frmMain: TfrmMain;

implementation

{$R *.lfm}

function pack_letter(s : string) : string;
var
  i : integer;
  r : string;
  c : char;
begin
  r := '';
  for i := 1 to length(s) do
  begin
    c := s[i];
    if pos(c,r) > 0 then continue;
    r := r + c;
  end;
  result := r;
end;

function compare_rate(s,t : string; related_to_s : boolean = True) : double;
var
  use_packt : boolean;
  packs,packt,r,mais_r : string;
  positions : array of array of integer;
  maiores : array of array of string;
  i,j,k,l,p,m,n,maior,mais_maior : integer;
  c : char;

  function get_p_t(i,j : integer) : integer;
  var
    p : integer;
    c : char;
  begin
    c := t[i];
    p := pos(c,packt);
    result := positions[p,j];
  end;

begin
  if s = t
  then begin
    Result := 1;
    exit;
  end;
  if related_to_s and (length(s) = 0)
  then begin
    Result := 0;
    exit;
  end;
  if not related_to_s and (length(t) = 0)
  then begin
    Result := 0;
    exit;
  end;
  packs := pack_letter(s);
  packt := pack_letter(t);
  use_packt := length(s)*length(packt) <= length(t)*length(packs);
  if not use_packt
  then begin
    packt := packs;
    packs := s;
    s := t;
    t := packs;
  end;
  // inicializar variáveis "maiores" e "positions"
  SetLength(positions,Length(packt)+1,Length(s)+1);
  SetLength(maiores,Length(t)+1,Length(s)+1);

  mais_maior := 0;
  mais_r := '';
  for i := 1 to Length(s) do
  begin
    c := s[i];
    p := pos(c,packt);
    if p <= 0 then Continue;
    for j := 1 to length(s) do
    begin
      if positions[p,j] = 0
      then begin
        positions[p,j] := i;
        break;
      end;
    end;
  end;

  for i := length(t) downto 1 do
  begin
    for j := 1 to length(s) do
    begin
      m := get_p_t(i,j);
      if m = 0 then break;
      maior := 0;
      r := '';
      for k := i+1 to length(t) do
      begin
        for l := 1 to length(s) do
        begin
          n := get_p_t(k,l);
          if n = 0 then break;
          if (n > m) and (length(maiores[k,l]) > maior)
          then begin
            r := maiores[k,l];
            maior := length(r);
          end;
        end;
      end;
      r := t[i] + r;
      maiores[i,j] := r;
      if length(r) > mais_maior
      then begin
        mais_maior := length(r);
        mais_r := r;
      end;
    end;
  end;
  r := mais_r;
  maior := mais_maior;
  //ShowMessage('String semelhante = "' + r + '"');
  //r := '';
  //for i := 1 to contador do
  //  if i = 1 then r := conjunto[i] else r := r + #10 + conjunto[i];
  //showmessage(r);
  if not use_packt
  then begin
    packs := s;
    s := t;
    t := packs;
  end;
  if related_to_s then Result := maior / length(s) else Result := maior / length(t);
  if Result = 1 then Result := 0.999;
end;

function MyRunProcess(comando : string; var saida : string) : integer;
var
  RetCode : integer;
  AProcess : TProcess; // adicionar "process" a Uses
  AStringList : TStringList;
begin
  Screen.Cursor := crHourGlass;
  AProcess := Nil;
  AStringList := Nil;
  Result := -1;
  try
    AProcess := TProcess.Create(nil);
    AProcess.CommandLine := comando;
    //AProcess.Executable := FindDefaultExecutablePath('python');
    //AProcess.Parameters.Add(ScriptName);
    //if parametros <> ''
    //then
    //  AProcess.Parameters.Add(parametros);
    AProcess.Options := AProcess.Options + [poWaitOnExit, poUsePipes];
    AProcess.Execute;

    AStringList := TStringList.Create;
    AStringList.LoadFromStream(AProcess.Output);
    RetCode := AProcess.ExitCode;
    Result := RetCode;
    saida := AStringList.Text;
  except end;
  try FreeAndNil(AStringList) except end;
  try FreeAndNil(AProcess) except end;
  Screen.Cursor := crDefault;
end;

function RunPythonScript(ScriptName, parametros : string; var saida : string) : integer;
var
  RetCode : integer;
  AProcess : TProcess; // adicionar "process" a Uses
  AStringList : TStringList;
begin
  Screen.Cursor := crHourGlass;
  AProcess := Nil;
  AStringList := Nil;
  Result := -1;
  try
    AProcess := TProcess.Create(nil);
    AProcess.Executable := FindDefaultExecutablePath('python');
    AProcess.Parameters.Add(ScriptName);
    if parametros <> ''
    then
      AProcess.Parameters.Add(parametros);
    AProcess.Options := AProcess.Options + [poWaitOnExit, poUsePipes];
    AProcess.Execute;

    AStringList := TStringList.Create;
    AStringList.LoadFromStream(AProcess.Output);
    RetCode := AProcess.ExitCode;
    Result := RetCode;
    saida := AStringList.Text;
  except end;
  try FreeAndNil(AStringList) except end;
  try FreeAndNil(AProcess) except end;
  Screen.Cursor := crDefault;
end;

function MakeURLParam(Key,Value : string) : string;
begin
  Result := EncodeURLElement(Key)+'='+EncodeURLElement(Value);
end;

procedure SeparaCores(C : TColor; var R,G,B : Integer);
begin
  R := (C and 255);
  G := ((C shr 8) and 255);
  B := ((C shr 16) and 255);
end;

function LerImagem(NomeArq : string) : TBitmap;
var
  IMG : TImage;
  newImg : TBitmap;
  R : TRect;
  //X,Y,cr,cg,cb,cm : integer;
  //cor : TColor;
begin
  IMG := TImage.Create(nil);
  try
    IMG.Picture.LoadFromFile(NomeArq);
    newImg := TBitmap.Create;
    newImg.Width := IMG.Picture.Width;
    newImg.Height := IMG.Picture.Height;
    R := Rect(0,0,IMG.Picture.Width,IMG.Picture.Height);
    newImg.Canvas.CopyRect(R,IMG.Picture.Bitmap.Canvas,R);
    {
    for X := 0 to newImg.Width - 1 do
      for Y := 0 to newImg.Height - 1 do
      begin
        cor := newImg.Canvas.Pixels[X,Y];
        SeparaCores(cor,cr,cg,cb);
        cm := (cr+cg+cb) div 3;
        cm := Round(cm + (255-cm)*0.75);
        newImg.Canvas.Pixels[X,Y] := RGBToColor(cm,cm,cm);
      end;
    }
    Result := newImg;
  finally
    IMG.Free;
  end;
end;

function GetValueJSON(No : TJSONData; Key : string) : String;
var
  SubNo : TJSONData;
begin
  SubNo := No.FindPath(Key);
  if SubNo = nil then Result := '' else Result := SubNo.AsString;
end;

function GetNodeJSON(No : TJSONData; Key : string) : TJSONData;
begin
  Result := No.FindPath(Key);
end;

procedure SetValueJSON(No : TJSONData; Key : string; Valor : Variant);
var
  SubNo : TJSONData;
begin
  SubNo := No.FindPath(Key);
  if SubNo <> nil then SubNo.Value := Valor;
end;

function GetJsonValores(ImgFilename: string; var Dados: TJSONData) : boolean;
var
  NomeValores : string;
  FT : TextFile;
  Linha,Linhas : string;
begin
  Dados := GetJSON('{}');
  Result := False;
  NomeValores := ChangeFileExt(ImgFilename,'.json');
  if FileExists(NomeValores)
  then begin
    try
      try
        AssignFile(FT,NomeValores);
        Reset(FT);
        Linhas := '';
        while not Eof(FT) do
        begin
          Readln(FT,Linha);
          Linhas := Linhas + Linha + ' ';
        end;
        Dados := GetJSON(Linhas);
        Result := True;
      finally
        CloseFile(FT);
      end;
    except
      Dados := GetJSON('{}');
    end;
  end;
end;

procedure TfrmMain.ClearGrid;
var
  i,j : integer;
begin
  for i := 1 to grdValores.RowCount - 1 do
    for j := 0 to grdValores.ColCount - 1 do
      grdValores.Cells[j,i] := '';
  grdValores.RowCount := 1;
end;

function TfrmMain.FindGridRow(Chave : string) : integer;
var
  i : integer;
begin
  for i := 1 to grdValores.RowCount - 1 do
    if Chave = grdValores.Cells[0,i]
    then begin
      Result := i;
      exit;
    end;
  Result := -1;
end;

procedure TfrmMain.AddLine(Dados: TJSONData; Key : string);
var
  V : string;
begin
  try
    V := Dados.FindPath(Key).AsString;
  except
    Exit;
  end;
  if V = '' then Exit;
  lblStatus.Caption := UpperCase(Key+': "'+V+'"');
end;

procedure TfrmMain.MakeAutentikusHeader(var HTTPObj : TFPHTTPClient);
var
  ExpTime : integer;
  Saida,url,auth_basic,comandoCURL : string;
  jsondata : TJSONData;
  env_auth : string;
  param_ambiente : string;

begin
  if Ignore_Autentikus then Exit;

  if Use_ValAutentikus
  then begin
    auth_basic := 'dWhqZmxxYTYwZWVjN29maGU4NDNvOXBjbTA6bTFzZ3B1ZzUwczlnbDJrdGc1Y2NnaDVobnI=';
    url := 'valautentikus';
    env_auth := '2';
    param_ambiente := 'D'
  end
  else begin
    auth_basic := 'M2MzZWVtMjE0dGk3Zmw1MTJvNzFjcGVrMTE6OXZmNWg5aDRocGk4ZGFmc3Rua2psaXFiN2Q=';
    url := 'autentikus';
    env_auth := '1';
    param_ambiente := 'P'
  end;
  //comandoCURL := 'curl --request POST'
  //  + ' --url https://' + url + '.estaleiro.serpro.gov.br/autentikus-authn/api/v1/token'
  //  + ' --header ''Authorization: Basic ' + auth_basic + ''''
  //  + ' --header ''content-type: application/x-www-form-urlencoded'''
  //  + ' --data ''grant_type=client_credentials&scope=escopo_vcdoc''';

  if not BearerValido
  then begin
    if RunPythonScript(PathProg + 'getbearer.py',param_ambiente,Saida) <> 0
    //if MyRunProcess(comandoCURL, Saida) <> 0
    then
      raise Exception.Create('Erro ao obter o token de autenticação');

    jsondata := GetJSON(saida);
    ExpTime := StrToIntDef(GetValueJSON(jsondata,'expires_in'), -1);
    if ExpTime = -1
    then
      raise Exception.Create('Erro ao obter o token de autenticação');
    if ExpTime >= 5 then Dec(ExpTime,5) else ExpTime := 0;
    BearerValido := True;
    BearerAtual := GetValueJSON(jsondata,'access_token');
    tmrBearerValido.Enabled := True;
    tmrBearerValido.Interval := ExpTime * 1000 + 200;
  end;

  //memJSON.Text:= '"' + BearerAtual + '"';
  //HTTPObj.AddHeader('User-Agent','Mozilla 5.0 (compatible ct)');
  //HTTPObj.AddHeader('content-type','application/json;charset=UTF-8');
  HTTPObj.AddHeader('Authorization','Bearer ' + BearerAtual);
  HTTPObj.AddHeader('typ', 'JWT');
  HTTPObj.AddHeader('alg', 'RS512');
  HTTPObj.AddHeader('env_auth', env_auth);
end;

procedure TfrmMain.RunApi(ApiURI : string; Estilo : TAPYStyle);
var
  Respo : TStringStream;
  S, Simg, ObjName, OcrText : string;
  Score : Double;
  Xmin,Ymin,Wid,Hei : integer;
  Xmax,Ymax : integer;
  ImgFile : File of Char;
  Ch : Char;
  Dados,RL,RLI,BBox : TJSONData;
  i : integer;
  OldCursor : TCursor;
  HTTPSock : TFPHttpClient;
  IsCNH,IsRG,TemCampoCNHVerso : Boolean;
  BMP : TBitmap;
  TipoDoc,Chave : string;
  KeyInd,NumMatch : integer;
  V,MatchSum : Double;
  sResize,sRotation : string;

  procedure VerifiqueTipoDocFromRL;
  var
    i : integer;
  begin
    IsCNH := False;
    IsRG := False;
    TemCampoCNHVerso := False;
    for i := 0 to RL.Count-1 do
    begin
      RLI := RL.Items[i];
      ObjName := GetValueJSON(RLI,'obj_name');
      if (LowerCase(LeftStr(ObjName,3)) = 'cnh')
      or (LowerCase(RightStr(ObjName,3)) = 'cnh')
      then
        IsCNH := True;
      if (LowerCase(LeftStr(ObjName,2)) = 'rg')
      or (LowerCase(RightStr(ObjName,2)) = 'rg')
      then
        IsRG := True;
      if (LowerCase(ObjName) = 'local_emissao_cnh')
      or (LowerCase(ObjName) = 'data_emissao_cnh')
      then
        TemCampoCNHVerso := True;
    end;
    if IsCNH <> IsRG
    then begin
      if IsCNH
      then begin
        if TemCampoCNHVerso
        then
          grpDoc.ItemIndex := 1
        else
          grpDoc.ItemIndex := 2;
      end
      else
        grpDoc.ItemIndex := 3;
    end;
  end;

begin
  if lstArquivos.Count = 0 then Exit;
  OldCursor := Screen.Cursor;
  Screen.Cursor := crHourGlass;
  LimpaTudo;
  try
    grpDoc.Enabled := False;
    grpDoc.ItemIndex := -1;
    grdValores.Enabled := False;
    ClearGrid;
    AssignFile(ImgFile,edtImagem.Text);
    Reset(ImgFile);
    Simg := '';
    while not Eof(ImgFile) do
    begin
      Read(ImgFile,Ch);
      Simg := Simg + Ch;
    end;
    CloseFile(ImgFile);

    if chkRotation.Checked
    then sRotation := '1'
    else sRotation := '0';

    if chkResizing.Checked
    then sResize := IntToStr(StrToIntDef(edtMaxLarg.Text,800))
    else sResize := '0';
    S := EncodeStringBase64(Simg);
    S := MakeURLParam('image',S);
    HTTPSock := TFPHttpClient.Create(Nil);
    MakeAutentikusHeader(HTTPSock);

    HTTPSock.AddHeader('requester','vcservertest');
    HTTPSock.AddHeader('vcdoc_param_avaliar_rotacao',sRotation);
    HTTPSock.AddHeader('vcdoc_param_largura_de_redimensionamento_imagem',sResize);
    HTTPSock.AddHeader('vcdoc_param_crop_height','120');
    HTTPSock.AddHeader('vcdoc_param_ocr_solicitado','2');

    With HTTPSock do
      try
        try
          Respo := TStringStream.Create('');
          FormPost(ApiURI,S,Respo);
          S := Respo.DataString;
          Dados := GetJSON(S);
          SetValueJSON(Dados,'pred_img','');
          SetValueJSON(Dados,'work_img','');
          memJSON.Lines.Text := Dados.FormatJSON;
          AddLine(Dados,'status');
          case Estilo of
          asDetect, asDetectOCR:
            begin
              grpDoc.Enabled := True;
              grpDoc.ItemIndex := 0;
              grdValores.Enabled := True;
              ClearGrid;
              lblPercentMatch.Caption := ' ';
              RL := Dados.FindPath('resultlist');
              if RL = nil then Exit;
              if not GetJsonValores(edtImagem.Text,Valores)
              then begin
                VerifiqueTipoDocFromRL;
              end
              else begin
                TipoDoc := GetValueJSON(Valores,'tipo_doc');
                if TipoDoc = ''
                then begin
                  VerifiqueTipoDocFromRL;
                end
                else begin
                  if LowerCase(TipoDoc) = 'cnh' then grpDoc.ItemIndex := 1
                  else if LowerCase(TipoDoc) = 'cnh_frente' then grpDoc.ItemIndex := 2
                  else if LowerCase(LeftStr(TipoDoc,2)) = 'rg' then grpDoc.ItemIndex := 3
                  else grpDoc.ItemIndex := 0;
                end;
                for i := 1 to grdValores.RowCount-1 do
                begin
                  Chave := grdValores.Cells[0,i];
                  if Chave = '' then Continue;
                  grdValores.Cells[4,i] := AnsiUpperCase( GetValueJSON(Valores,Chave) );
                end;
              end;
              MatchSum := 0;
              NumMatch := 0;
              if RL.Count > 0
              then begin
                try
                  BMP := LerImagem(edtImagem.Text);
                  for i := 0 to RL.Count-1 do
                  begin
                    RLI := RL.Items[i];
                    BBox := RLI.FindPath('bounding_box');
                    Xmin := GetNodeJSON(BBox,'x_min').AsInteger;
                    Ymin := GetNodeJSON(BBox,'y_min').AsInteger;
                    Wid := GetNodeJSON(BBox,'width').AsInteger;
                    Hei := GetNodeJSON(BBox,'height').AsInteger;
                    Xmax := Xmin + Wid;
                    Ymax := Ymin + Hei;
                    ObjName := GetValueJSON(RLI,'obj_name');
                    OcrText := AnsiUpperCase(GetValueJSON(RLI,'adjusted_ocr'));
                    //GetNodeJSON(RLI,'candidates_ocr').
                    Score := GetNodeJSON(RLI,'score').AsFloat * 100;

                    KeyInd := FindGridRow(ObjName);
                    if KeyInd > 0
                    then begin
                      if (grdValores.Cells[4,KeyInd] = '')
                      then begin
                        grdValores.Cells[4,KeyInd] := OcrText;
                        V := -1;
                      end
                      else begin
                        V := compare_rate(grdValores.Cells[4,KeyInd],OcrText);
                        MatchSum := MatchSum + V;
                        Inc(NumMatch);
                      end;
                    end
                    else begin
                      V := -1;
                      KeyInd := grdValores.RowCount;
                      grdValores.RowCount := KeyInd + 1;
                      grdValores.Cells[0,KeyInd] := ObjName;
                      grdValores.Cells[4,KeyInd] := OcrText;
                    end;
                    grdValores.Cells[3,KeyInd] := OcrText;
                    if V <> -1
                    then grdValores.Cells[2,KeyInd] := FormatFloat('##0.##',V*100) + '%'
                    else  grdValores.Cells[2,KeyInd] := 'N/A';
                    grdValores.Cells[1,KeyInd] := FormatFloat('##0.##',Score) + '%';

                    if Xmin < 0 then Xmin := 0;
                    if Ymin < 0 then Ymin := 0;
                    if Xmax >= BMP.Width then Xmax := BMP.Width-1;
                    if Ymax >= BMP.Height then Ymax := BMP.Height-1;
                    BMP.Canvas.Pen.Color := clBlue;
                    BMP.Canvas.Pen.Width := 2;
                    BMP.Canvas.Line(Xmin,Ymin,Xmin,Ymax);
                    BMP.Canvas.Line(Xmin,Ymax,Xmax,Ymax);
                    BMP.Canvas.Line(Xmax,Ymax,Xmax,Ymin);
                    BMP.Canvas.Line(Xmax,Ymin,Xmin,Ymin);
                    BMP.Canvas.Pixels[Xmin,Ymin] := clLime;
                    BMP.Canvas.Pixels[Xmin+1,Ymin] := clLime;
                    BMP.Canvas.Pixels[Xmin+1,Ymin+1] := clLime;
                    BMP.Canvas.Pixels[Xmin,Ymin+1] := clLime;
                  end;
                  if NumMatch > 0
                  then begin
                    lblPercentMatch.Caption := 'Percentual de Acerto dessa Imagem: ' + FormatFloat('##0.##',MatchSum * 100 / NumMatch) + '%';
                  end;
                  BMP.SaveToFile(PathProg+'___temp___.bmp');
                  IMG.Picture.LoadFromFile(PathProg+'___temp___.bmp');
                  DeleteFile(PathProg+'___temp___.bmp');
                finally
                  BMP.Free;
                end;
              end;
            end;
          end;
          Respo.Destroy;
        except
          ShowMessage('Erro ao acessar o VCDOC Server');
        end;
      finally
        Free;
      end;
  finally
    Screen.Cursor := OldCursor;
  end;
end;

procedure TfrmMain.SelecionaBaseAddress;
begin
  case ApiEnvironment[1] of
  'P': begin ApiAddress := AddressP; Ignore_Autentikus := False; Use_ValAutentikus := False; end;
  'H': begin ApiAddress := AddressH; Ignore_Autentikus := False; Use_ValAutentikus := True;  end;
  'D': begin ApiAddress := AddressD; Ignore_Autentikus := False; Use_ValAutentikus := True;  end;
  'L': begin ApiAddress := AddressL; Ignore_Autentikus := True;  Use_ValAutentikus := True;  end;
  end;
  if (ApiAddress <> '') and (ApiAddress[Length(ApiAddress)] <> '/')
  then
    ApiAddress := ApiAddress + '/';
  lblBaseAddress.Caption := 'Endereço Base: "' + ApiAddress + '"';
end;

procedure TfrmMain.btnDetectOCRClick(Sender: TObject);
begin
  RunApi(ApiAddress+ApiDetectOCR,asDetectOCR);
end;

procedure TfrmMain.chkResizingChange(Sender: TObject);
begin
  edtMaxLarg.Enabled := chkResizing.Checked;
end;

procedure TfrmMain.btnDetectClick(Sender: TObject);
//var
//  s,t : string;
//  d : double;
begin
  //s := 'alexandre';
  //t := 'a.l.dr.e.x.a.d.r.e';
  //d := compare_rate(s,t);
  //ShowMessage('Comparação:' + FloatToStr(d*100));
  //exit;

  RunApi(ApiAddress+ApiDetect,asDetect);
end;

procedure TfrmMain.edtDirChange(Sender: TObject);
begin
  lstArquivos.Directory := edtDir.Directory;
  lstArquivos.ItemIndex := -1;
  LimpaTudo(True);
  ImagesDir := IncludeTrailingPathDelimiter(edtDir.Directory);
end;

procedure TfrmMain.edtMaxLargEditingDone(Sender: TObject);
var
  S : string;
  i : integer;
  ch : char;
begin
  S := '';
  for i := 1 to Length(edtMaxLarg.Text) do
  begin
    ch := edtMaxLarg.Text[i];
    if ch in ['0'..'9'] then S := S + ch;
  end;
  edtMaxLarg.Text := S;
  i := StrToIntDef(S,-1);
  if i < 0
  then begin
    ShowMessage('Valor inválido para a largura máxima');
    Exit;
  end;
  if i < 480
  then begin
    ShowMessage('A largura máxima deve ser maior que 480');
    edtMaxLarg.Text := '480';
  end;
end;

procedure TfrmMain.FormActivate(Sender: TObject);
var
  ini : TIniFile;
begin
  ini := TIniFile.Create(PathProg + 'api1.ini');
  try
    Redimensionando := True;
    // -------- Geometry
    FileListHeight := ini.ReadFloat('Geometry','FileListHeight',1/3);
    ImagePanelWidth := ini.ReadFloat('Geometry','ImagePanelWidth',1/4);
    JSONPanelHeight := ini.ReadFloat('Geometry','JSONPanelHeight',1/4);
    ColGridValoresWidth0 :=  ini.ReadFloat('Geometry','ColGridValoresWidth0',1/6);
    ColGridValoresWidth1 :=  ini.ReadFloat('Geometry','ColGridValoresWidth1',1/7);
    ColGridValoresWidth2 :=  ini.ReadFloat('Geometry','ColGridValoresWidth2',1/7);
    ColGridValoresWidth3 :=  ini.ReadFloat('Geometry','ColGridValoresWidth3',1/6);
  finally
    Redimensionando := False;
    Comecou := True;
    IMG.OnPictureChanged := @IMGPictureChanged;
    ini.Free;
    FormResize(nil);
  end;
end;

procedure TfrmMain.FormClose(Sender: TObject; var CloseAction: TCloseAction);
var
  ini : TIniFile;
begin
  ini := TIniFile.Create(PathProg + 'api1.ini');
  try
    // -------- Files
    ini.WriteString('Files','ImagesDir',edtDir.Directory);

    // -------- APIServer
    ini.WriteString('APIServer','Environment',ApiEnvironment);

    // -------- Geometry
    if JSONPanelHeight > 0.25 then JSONPanelHeight := 0.25;
    if ImagePanelWidth > 0.25 then ImagePanelWidth := 0.25;
    ini.WriteInteger('Geometry','WindowState',Ord(WindowState));
    ini.WriteInteger('Geometry','FormWidth',Width);
    ini.WriteInteger('Geometry','FormHeight',Height);
    ini.WriteFloat('Geometry','FileListHeight',FileListHeight);
    ini.WriteFloat('Geometry','ImagePanelWidth',ImagePanelWidth);
    ini.WriteFloat('Geometry','JSONPanelHeight',JSONPanelHeight);
    ini.WriteFloat('Geometry','ColGridValoresWidth0',ColGridValoresWidth0);
    ini.WriteFloat('Geometry','ColGridValoresWidth1',ColGridValoresWidth1);
    ini.WriteFloat('Geometry','ColGridValoresWidth2',ColGridValoresWidth2);
    ini.WriteFloat('Geometry','ColGridValoresWidth3',ColGridValoresWidth3);

    // -------- Detect
    ini.WriteBool('Detect','UseRotation',chkRotation.Checked);
    ini.WriteBool('Detect','UseResizing',chkResizing.Checked);
    ini.WriteInteger('Detect','MaxWidth',StrToIntDef(edtMaxLarg.Text,800));
  finally
    ini.Free;
  end;
end;

procedure TfrmMain.FormCreate(Sender: TObject);
var
  ini : TIniFile;
begin
  BearerValido := False;
  PathProg := IncludeTrailingPathDelimiter(ExtractFilePath(ExpandFileName(Application.ExeName)));
  ini := TIniFile.Create(PathProg + 'api1.ini');
  try
    // -------- Files
    edtDir.Directory := ini.ReadString('Files','ImagesDir',PathProg);

    // -------- APIServer
    ApiEnvironment := ini.ReadString('APIServer','Environment','D');
    if ApiEnvironment = '' then ApiEnvironment := 'D';
    ApiEnvironment := Copy(ApiEnvironment,1,1);
    AddressP := ini.ReadString('APIServer','AddressP','https://vcdoc-vcdocgpu.p02.estaleiro.serpro.gov.br/v1/');
    AddressD := ini.ReadString('APIServer','AddressD','https://vcdoc-dev-vcdocgpu.p02.estaleiro.serpro.gov.br/v1/');
    AddressH := ini.ReadString('APIServer','AddressH','https://vcdoc-hom-vcdocgpu.p02.estaleiro.serpro.gov.br/v1/');
    AddressL := ini.ReadString('APIServer','AddressL','http://localhost:8081/v1/');
    ApiDetect    := ini.ReadString('APIServer','Detect',     'detect');
    ApiDetectOCR := ini.ReadString('APIServer','DetectOCR',  'detect-ocr');
    Valores := GetJSON('{}');
    case ApiEnvironment[1] of
    'P': tabAmbientes.TabIndex := 0;
    'H': tabAmbientes.TabIndex := 1;
    'D': tabAmbientes.TabIndex := 2;
    'L': tabAmbientes.TabIndex := 3;
    end;
    SelecionaBaseAddress;

    // -------- Geometry
    FileListHeight := ini.ReadFloat('Geometry','FileListHeight',1/3);
    Width := ini.ReadInteger('Geometry','FormWidth',Round(Screen.Width * 0.75));
    Height := ini.ReadInteger('Geometry','FormHeight',Round(Screen.Height * 0.75));
    WindowState := TWindowState(ini.ReadInteger('Geometry','WindowState',Ord(wsMaximized)));

    // -------- Detect
    chkRotation.Checked := ini.ReadBool('Detect','UseRotation',False);
    chkResizing.Checked := ini.ReadBool('Detect','UseResizing',True);
    edtMaxLarg.Enabled := chkResizing.Checked;
    edtMaxLarg.Text := IntToStr(ini.ReadInteger('Detect','MaxWidth',800));
  finally
    ini.Free;
  end;
  grdValores.DefaultRowHeight := 36;
  grdValores.Cells[0,0] := 'Campo';
  grdValores.Cells[1,0] := 'Score YOLO';
  grdValores.Cells[2,0] := '%Acerto OCR';
  grdValores.Cells[3,0] := 'Valor Detectado';
  grdValores.Cells[4,0] := 'Valor Esperado';
end;

procedure TfrmMain.FormResize(Sender: TObject);
begin
  if not Comecou then Exit;
  Redimensionando := True;
  try
    lstArquivos.Height := Round(FileListHeight * panInfos.Height);
    panImagem.Width := Round(ImagePanelWidth * Width);
    panJSON.Height := Round(JSONPanelHeight * panInfos.Height);
  finally
    Redimensionando := False;
  end;
end;

procedure TfrmMain.grdValoresHeaderSized(Sender: TObject; IsColumn: Boolean;
  Index: Integer);
begin
  ColGridValoresWidth0 := grdValores.ColWidths[0] / grdValores.Width;
  ColGridValoresWidth1 := grdValores.ColWidths[1] / grdValores.Width;
  ColGridValoresWidth2 := grdValores.ColWidths[2] / grdValores.Width;
  ColGridValoresWidth3 := grdValores.ColWidths[3] / grdValores.Width;
  if Index < 4
  then begin
    with grdValores do
      ColWidths[4] := Width - 16 - ColWidths[0] - ColWidths[1] - ColWidths[2] - ColWidths[3];
  end;
end;

procedure TfrmMain.grdValoresResize(Sender: TObject);
begin
  with grdValores do
  begin
    ColWidths[0] := Round(ColGridValoresWidth0 * Width);
    ColWidths[1] := Round(ColGridValoresWidth1 * Width);
    ColWidths[2] := Round(ColGridValoresWidth2 * Width);
    ColWidths[3] := Round(ColGridValoresWidth3 * Width);
    ColWidths[4] := Width - 16 - ColWidths[0] - ColWidths[1] - ColWidths[2] - ColWidths[3];
  end;
end;

procedure TfrmMain.grpDocClick(Sender: TObject);
begin
  with grpDoc do
  begin
    ClearGrid;
    grdValores.RowCount := 1;
    if ItemIndex <= 0
    then
      Exit;

    if (Items[ItemIndex] = 'CNH') or (Items[ItemIndex] = 'CNH Frente')
    then begin
      grdValores.RowCount := 10;
      grdValores.Cells[0,1] := 'nome_cnh';
      grdValores.Cells[0,2] := 'identidade_cnh';
      grdValores.Cells[0,3] := 'cpf_cnh';
      grdValores.Cells[0,4] := 'nascimento_cnh';
      grdValores.Cells[0,5] := 'filiacao_cnh';
      grdValores.Cells[0,6] := 'categoria_cnh';
      grdValores.Cells[0,7] := 'registro_cnh';
      grdValores.Cells[0,8] := 'validade_cnh';
      grdValores.Cells[0,9] := 'pri_habilitacao_cnh';

      if Items[ItemIndex] = 'cnh'
      then begin
        grdValores.RowCount := 13;
        grdValores.Cells[0,10] := 'observacao_cnh';
        grdValores.Cells[0,11] := 'local_emissao_cnh';
        grdValores.Cells[0,12] := 'data_emissao_cnh';
      end;
    end;
    if Items[ItemIndex] = 'RG'
    then begin
      grdValores.RowCount := 10;
      grdValores.Cells[0,1] := 'nome_rg';
      grdValores.Cells[0,2] := 'registro_geral_rg';
      grdValores.Cells[0,3] := 'data_expedicao_rg';
      grdValores.Cells[0,4] := 'filiacao_rg';
      grdValores.Cells[0,5] := 'naturalidade_rg';
      grdValores.Cells[0,6] := 'nascimento_rg';
      grdValores.Cells[0,7] := 'doc_origem_rg';
      grdValores.Cells[0,8] := 'cpf_rg';
      grdValores.Cells[0,9] := 'cabecalho_rg';
    end;
  end;
end;

procedure TfrmMain.IMGPictureChanged(Sender: TObject);
begin
  try
    if IMG = nil then exit;
    if IMG.Picture = nil then exit;
    Label1.Caption := '(L:' + IntToStr(IMG.Picture.Width) + ' x A:' + IntToStr(IMG.Picture.Height) + ')';
  except
    Label1.Caption := ' ';
  end;
end;

procedure TfrmMain.lstArquivosChange(Sender: TObject);
const
  NUM_CHAVES = 29;
  chaves : array [1..NUM_CHAVES] of string = (
  'foto_cnh',
  'nome_cnh',
  'identidade_cnh',
  'cpf_cnh',
  'nascimento_cnh',
  'filiacao_cnh',
  'registro_cnh',
  'validade_cnh',
  'pri_habilitacao_cnh',
  'local_emissao_cnh',
  'data_emissao_cnh',
  'cnh',
  'nome_rg',
  'foto_rg',
  'assinatura_rg',
  'digital_rg',
  'registro_geral_rg',
  'data_expedicao_rg',
  'filiacao_rg',
  'naturalidade_rg',
  'nascimento_rg',
  'doc_origem_rg',
  'cpf_rg',
  'rg_verso',
  'rg_frente',
  'cnh_frente',
  'categoria_cnh',
  'observacao_cnh',
  'cabecalho_rg'
  );
var
  IsCNH, IsRG, TemCampoCNHVerso : Boolean;
  TipoDoc,ObjName : string;
  i : integer;
begin
  if lstArquivos.ItemIndex = -1
  then begin
    edtImagem.Text := '';
    LimpaTudo(True);
  end
  else begin
    IMG.Picture.LoadFromFile(lstArquivos.FileName);
    if edtImagem.Text <> lstArquivos.FileName
    then
      edtImagem.Text := lstArquivos.FileName;

    lblStatus.Caption := ' ';
    lblPercentMatch.Caption := ' ';
    memJSON.Clear;
    if GetJsonValores(lstArquivos.FileName,Valores)
    then begin
      TipoDoc := GetValueJSON(Valores,'tipo_doc');
      if TipoDoc = ''
      then begin
        IsCNH := False;
        IsRG := False;
        TemCampoCNHVerso := False;
        for i := 1 to NUM_CHAVES do
        begin
          ObjName := chaves[i];
          if Valores.FindPath(ObjName) = nil then Continue;
          if (LowerCase(LeftStr(ObjName,3)) = 'cnh')
          or (LowerCase(RightStr(ObjName,3)) = 'cnh')
          then
            IsCNH := True;
          if (LowerCase(LeftStr(ObjName,2)) = 'rg')
          or (LowerCase(RightStr(ObjName,2)) = 'rg')
          then
            IsRG := True;
          if (LowerCase(ObjName) = 'local_emissao_cnh')
          or (LowerCase(ObjName) = 'data_emissao_cnh')
          then
            TemCampoCNHVerso := True;
        end;
        if IsCNH <> IsRG
        then begin
          if IsCNH
          then begin
            if TemCampoCNHVerso
            then grpDoc.ItemIndex := 1
            else grpDoc.ItemIndex := 2;
          end
          else
            grpDoc.ItemIndex := 3;
        end;
      end
      else begin
        if LowerCase(TipoDoc) = 'cnh' then grpDoc.ItemIndex := 1
        else if LowerCase(TipoDoc) = 'cnh_frente' then grpDoc.ItemIndex := 2
        else if LowerCase(LeftStr(TipoDoc,2)) = 'rg' then grpDoc.ItemIndex := 3
        else grpDoc.ItemIndex := 0;
      end;
      for i := 1 to grdValores.RowCount-1 do
      begin
        grdValores.Cells[1,i] := '';
        grdValores.Cells[2,i] := '';
        grdValores.Cells[3,i] := '';
        grdValores.Cells[4,i] := '';
        ObjName := grdValores.Cells[0,i];
        if ObjName = '' then Continue;
        grdValores.Cells[4,i] := AnsiUpperCase(GetValueJSON(Valores,ObjName));
      end;
    end
    else begin
      grpDoc.ItemIndex := -1;
    end;
  end;
end;

function TfrmMain.CreateJsonFromValores : string;
var
  TipoDoc,J,Chave,V : string;
  i : integer;
  json : TJSONData;
begin
  case grpDoc.ItemIndex of
  1: TipoDoc := 'cnh';
  2: TipoDoc := 'cnh_frente';
  3: if FindGridRow('rg_frente') >= 0 then TipoDoc := 'rg_frente'
     else if FindGridRow('rg_verso') >= 0 then TipoDoc := 'rg_verso'
     else TipoDoc := 'rg';
  else TipoDoc := 'outros';
  end;
  if TipoDoc = 'outros'
  then begin
    Result := '{"tipo_doc": "outros"}';
    Exit;
  end;
  J := '{"tipo_doc": "' + TipoDoc + '"';
  for i := 1 to grdValores.RowCount - 1 do
  begin
    Chave := grdValores.Cells[0,i];
    if (TipoDoc = 'rg_frente')
    and (
      (Chave = 'rg_frente') or
      (Chave = 'foto_rg') or
      (Chave = 'assinatura_rg') or
      (Chave = 'digital_rg')
      )
    then else
    if (TipoDoc = 'cnh_frente')
    and (
      (Chave = 'foto_cnh') or
      (Chave = 'cnh_frente')
      )
    then else
    if (TipoDoc = 'cnh')
    and (
      (Chave = 'foto_cnh') or
      (Chave = 'cnh')
      )
    then else
    if grdValores.Cells[4,i] = '' then Continue;
    J := J + ', "' + Chave + '": ""'
  end;
  J := J + '}';
  json := GetJSON(J);
  for i := 1 to grdValores.RowCount - 1 do
  begin
    Chave := grdValores.Cells[0,i];
    V := grdValores.Cells[4,i];
    if V = '' then Continue;
    json.FindPath(Chave).AsString := V;
  end;
  J := json.FormatJSON;
  Result := J;
end;

procedure TfrmMain.grdValoresEnter(Sender: TObject);
begin
  OldJSON := CreateJsonFromValores;
  OldFilename := ChangeFileExt(edtImagem.Text,'.json');
end;

procedure TfrmMain.grdValoresExit(Sender: TObject);
var
  NewJSON : string;
  FT : TextFile;
begin
  NewJSON := CreateJsonFromValores;
  if NewJSON <> OldJSON
  then begin
    AssignFile(FT,OldFilename);
    Rewrite(FT);
    Write(FT,NewJSON);
    CloseFile(FT);
  end;
end;

procedure TfrmMain.grdValoresKeyPress(Sender: TObject; var Key: char);
var
  S : string;
begin
  S := AnsiUpperCase(Key);
  if S <> '' then Key := S[1];
end;

procedure TfrmMain.mnuAbrirDiretorioClick(Sender: TObject);
begin
  edtDir.RunDialog;
end;

procedure TfrmMain.mnuSairClick(Sender: TObject);
begin
  Close;
end;

procedure TfrmMain.mnuSaveAndBackClick(Sender: TObject);
var
  N : integer;
begin
  if lstArquivos.Count = 0 then Exit;
  N := lstArquivos.ItemIndex;
  if N < 0 then Exit;
  SaveJSON;
  Dec(N);
  if N < 0 then N := lstArquivos.Count-1;
  lstArquivos.ItemIndex := N;
  lstArquivos.Click;
  if (ActiveControl = grdValores) or (ActiveControl.Parent = grdValores)
  then begin
    ActiveControl := grpDoc;
    OldFilename := ChangeFileExt(lstArquivos.FileName,'.json');
  end;
  //lstArquivosChange(Sender);
end;

procedure TfrmMain.mnuSaveAndNextClick(Sender: TObject);
var
  N : integer;
begin
  if lstArquivos.Count = 0 then Exit;
  N := lstArquivos.ItemIndex;
  if N < 0 then Exit;
  SaveJSON;
  Inc(N);
  if N >= lstArquivos.Count then N := 0;
  lstArquivos.ItemIndex := N;
  lstArquivos.Click;
  if (ActiveControl = grdValores) or (ActiveControl.Parent = grdValores)
  then begin
    ActiveControl := grpDoc;
    OldFilename := ChangeFileExt(lstArquivos.FileName,'.json');
  end;
  //lstArquivosChange(Sender);
end;

procedure TfrmMain.SaveJSON;
var
  J,FN : string;
  FT : TextFile;
begin
  if (ActiveControl = grdValores) or (ActiveControl.Parent = grdValores)
  then begin
    lstArquivos.Focused;
    ActiveControl := lstArquivos;
    OldFilename := ChangeFileExt(edtImagem.Text,'.json');
  end;
  J := CreateJsonFromValores;
  FN := ChangeFileExt(edtImagem.Text,'.json');
  AssignFile(FT,FN);
  Rewrite(FT);
  Write(FT,J);
  CloseFile(FT);
end;

procedure TfrmMain.splArquivosChangeBounds(Sender: TObject);
begin
  if not Comecou or Redimensionando then Exit;
  FileListHeight := lstArquivos.Height / panInfos.Height;
end;

procedure TfrmMain.splLeftChangeBounds(Sender: TObject);
begin
  if not Comecou or Redimensionando then Exit;
  ImagePanelWidth := panImagem.Width / Width;
end;

procedure TfrmMain.splTopChangeBounds(Sender: TObject);
begin
  if not Comecou or Redimensionando then Exit;
  JSONPanelHeight := panJSON.Height / panInfos.Height;
end;

procedure TfrmMain.tabAmbientesChange(Sender: TObject);
begin
  case tabAmbientes.TabIndex of
  0: ApiEnvironment := 'P';
  1: ApiEnvironment := 'H';
  2: ApiEnvironment := 'D';
  3: ApiEnvironment := 'L';
  end;
  SelecionaBaseAddress;
end;

procedure TfrmMain.tmrBearerValidoTimer(Sender: TObject);
begin
  tmrBearerValido.Enabled := False;
  BearerValido := False;
end;

procedure TfrmMain.LimpaTudo(ImagemTambem: Boolean = False);
begin
  if ImagemTambem
  then begin
    IMG.Picture.Clear;
    ClearGrid;
    grpDoc.ItemIndex := -1;
  end;
  //memResultList.Clear;
  memJSON.Clear;
  Application.ProcessMessages;
end;

end.