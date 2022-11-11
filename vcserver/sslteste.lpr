{$mode objfpc}{$H+}
program sslteste;
Uses
  fphttpclient, fpopenssl;

Var
  S,URL : string;
  HTTP  : TFPHttpClient;
begin
  URL := ParamStr(1);
  if URL = '' then URL := 'https://www.google.com';
  writeln('GET:'+URL);
  writeln('-------');

  HTTP := TFPHttpClient.Create(nil);
  writeln('RESPONSE:');
  try
    S := HTTP.Get(URL);
    writeln(S);
  finally
    HTTP.free;
  end;
end.
