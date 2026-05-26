"""Sube una carpeta grande a un dataset de Hugging Face (reanudable).

Pensado para usarse con el bucket montado por gcsfuse, evitando staging en disco:

  gcsfuse --implicit-dirs TU_BUCKET /mnt/actas
  export HF_TOKEN=hf_xxx
  python migrar_a_hf.py --folder /mnt/actas --repo TU-USUARIO/actas-peru-2026

upload_large_folder es reanudable: si se corta, relanza el mismo comando.
"""
import argparse
import os

from huggingface_hub import HfApi


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--folder", required=True, help="carpeta a subir (p. ej. el mount de gcsfuse)")
    ap.add_argument("--repo", required=True, help="usuario/nombre del dataset")
    ap.add_argument("--private", action="store_true", help="crear el repo como privado")
    args = ap.parse_args()

    api = HfApi(token=os.environ["HF_TOKEN"])
    api.create_repo(args.repo, repo_type="dataset", exist_ok=True, private=args.private)
    api.upload_large_folder(repo_id=args.repo, repo_type="dataset",
                            folder_path=args.folder)
    print(f"HF: subida completa o reanudable en https://huggingface.co/datasets/{args.repo}")


if __name__ == "__main__":
    main()
