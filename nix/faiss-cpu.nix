{ lib
, buildPythonPackage
, fetchPypi
, setuptools
, wheel
, pkgs ? import <nixpkgs> {}
}:

buildPythonPackage rec {
  pname = "faiss_cpu";
  version = "1.10.0";

  src = fetchPypi {
    inherit pname version;
    hash = "sha256-W9ylVfJLwDb01n+KWk1syRuNISbU543klsojzNRuR50=";
  };

  # do not run tests
  doCheck = false;

  buildInputs = [
    pkgs.python311Packages.numpy
  ];

  propogatedBuildInputs = [
    pkgs.python311Packages.numpy
  ];

  # specific to buildPythonPackage, see its reference
  pyproject = true;
  build-system = [
    setuptools
    wheel
  ];
}

