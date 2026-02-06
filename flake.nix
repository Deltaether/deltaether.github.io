{
  description = "AI Art Detector - Rust WASM + TypeScript";

  inputs = {
    nixpkgs.url = "github:NixOS/nixpkgs/nixos-unstable";
    rust-overlay.url = "github:oxalica/rust-overlay";
    flake-utils.url = "github:numtide/flake-utils";
  };

  outputs = { self, nixpkgs, rust-overlay, flake-utils }:
    flake-utils.lib.eachDefaultSystem (system:
      let
        overlays = [ (import rust-overlay) ];
        pkgs = import nixpkgs { inherit system overlays; };
        
        rustToolchain = pkgs.rust-bin.stable.latest.default.override {
          targets = [ "wasm32-unknown-unknown" ];
        };
      in
      {
        devShells.default = pkgs.mkShell {
          buildInputs = with pkgs; [
            rustToolchain
            wasm-pack
            wasm-bindgen-cli
            nodejs_20
            nodePackages.npm
            nodePackages.typescript
            esbuild
          ];

          shellHook = ''
            echo "AI Art Detector dev environment"
            echo "Commands:"
            echo "  npm run build:wasm  - Build Rust to WASM"
            echo "  npm run build:ts    - Build TypeScript"
            echo "  npm run build       - Build all"
          '';
        };
      }
    );
}
