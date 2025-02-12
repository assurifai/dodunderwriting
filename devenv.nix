{ pkgs, lib, config, inputs, ... }:
let
  buildInputs = with pkgs; [
    stdenv.cc.cc
    libuv
    zlib
  ];
in 
{
  env = { LD_LIBRARY_PATH = "${with pkgs; lib.makeLibraryPath buildInputs}"; };

  languages.python = {
    enable = true;
    uv = {
      enable = true;
      sync.enable = true;
    };
  };

  scripts.start-app.exec = "streamlit run app.py";

  enterShell = ''
    . .devenv/state/venv/bin/activate
  '';
}
