let
  pkgs = import <nixpkgs> {};
  python = pkgs.python3.override {
    self = python;
    packageOverrides = pyfinal: pyprev: {
     finnhub-python = pyfinal.callPackage ./finnhub-python.nix { };
     fmpsdk = pyfinal.callPackage ./fmpsdk.nix { };
     pyxirr = pyfinal.callPackage ./pyxirr.nix { };
     faiss-cpu = pyfinal.callPackage ./faiss-cpu.nix { };
    };
  };
in pkgs.mkShell {
  packages = [
    (python.withPackages (python-pkgs: [
      python-pkgs.black
      python-pkgs.python-dotenv
      python-pkgs.pypdf2
      python-pkgs.pandas
      python-pkgs.faiss-cpu
      python-pkgs.openai
      python-pkgs.tiktoken
      python-pkgs.langchain
      python-pkgs.langchain-community
      python-pkgs.plotly
      python-pkgs.requests
      python-pkgs.streamlit
      python-pkgs.pyxirr
      python-pkgs.yfinance
      python-pkgs.numpy
      python-pkgs.finnhub-python
      python-pkgs.unidecode
      python-pkgs.requests-cache
    ]))
  ];
}
