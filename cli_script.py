import argparse
import pandas as pd
from main_script import build_full_pipeline
from pathlib import Path

def main():
    parser = argparse.ArgumentParser(
        description="Procesa un CSV de productos y genera dimensiones/peso imputados."
    )

    parser.add_argument(
        "--input",
        required=True,
        type=str,
        help="Ruta del archivo CSV de entrada."
    )

    parser.add_argument(
        "--output",
        required=True,
        type=str,
        help="Ruta del archivo CSV de salida expandido."
    )

    args = parser.parse_args()

    input_path = Path(args.input)
    output_path = Path(args.output)

    print(f"\n[CLI] Cargando archivo: {input_path}")
    df_final, models = build_full_pipeline(input_path)

    print(f"[CLI] Guardando resultado en: {output_path}")
    df_final.to_csv(output_path, index=False)

    print("[CLI] Proceso completado exitosamente ✔")

if __name__ == "__main__":
    main()
