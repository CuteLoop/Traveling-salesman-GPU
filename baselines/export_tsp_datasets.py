from pathlib import Path

import pandas as pd

BERLIN52_URL = "https://github.com/Valdecy/Datasets/raw/master/Combinatorial/TSP-02-Coordinates.txt"


def ensure_data_dir() -> Path:
    out = Path("data")
    out.mkdir(parents=True, exist_ok=True)
    return out


def write_tsplib(path: Path, name: str, comment: str, coords, edge_weight_type: str = "EUC_2D") -> None:
    lines = [
        f"NAME: {name}",
        "TYPE: TSP",
        f"COMMENT: {comment}",
        f"DIMENSION: {len(coords)}",
        f"EDGE_WEIGHT_TYPE: {edge_weight_type}",
        "NODE_COORD_SECTION",
    ]

    for i, (x, y) in enumerate(coords, start=1):
        lines.append(f"{i} {x:.10f} {y:.10f}")

    lines.append("EOF")
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def export_madeira(data_dir: Path) -> None:
    madeira_csv = data_dir / "madeira_cities.csv"
    if not madeira_csv.exists():
        raise FileNotFoundError(f"Missing expected file: {madeira_csv}")

    df = pd.read_csv(madeira_csv)

    required_cols = {"id", "city", "Lat", "Long"}
    if not required_cols.issubset(df.columns):
        raise ValueError(
            f"Madeira CSV columns must include {sorted(required_cols)}; got {list(df.columns)}"
        )

    # Keep CSV as canonical local copy (normalize order)
    df = df[["id", "city", "Lat", "Long"]].copy()
    df.to_csv(madeira_csv, index=False)

    # TSPLIB export uses the same numeric coordinates present in CSV.
    # NOTE: this is written as EUC_2D coordinates, matching how other project parsers consume x/y.
    coords = list(zip(df["Lat"].astype(float), df["Long"].astype(float)))
    write_tsplib(
        data_dir / "madeira_cities.tsp",
        name="MADEIRA_25",
        comment="Madeira cities exported from data/madeira_cities.csv",
        coords=coords,
        edge_weight_type="EUC_2D",
    )


def export_berlin52(data_dir: Path) -> None:
    df = pd.read_csv(BERLIN52_URL, sep="\t")
    if list(df.columns) != ["x", "y"]:
        raise ValueError(f"Unexpected Berlin52 schema: {list(df.columns)}")

    df_out = pd.DataFrame(
        {
            "id": range(1, len(df) + 1),
            "x": df["x"].astype(float),
            "y": df["y"].astype(float),
        }
    )
    df_out.to_csv(data_dir / "berlin52.csv", index=False)

    coords = list(zip(df_out["x"], df_out["y"]))
    write_tsplib(
        data_dir / "berlin52.tsp",
        name="BERLIN52",
        comment="Berlin52 exported from Valdecy TSP-02-Coordinates dataset",
        coords=coords,
        edge_weight_type="EUC_2D",
    )


def main() -> None:
    data_dir = ensure_data_dir()

    export_madeira(data_dir)
    export_berlin52(data_dir)

    print("Dataset export complete:")
    print(f"  - {data_dir / 'madeira_cities.csv'}")
    print(f"  - {data_dir / 'madeira_cities.tsp'}")
    print(f"  - {data_dir / 'berlin52.csv'}")
    print(f"  - {data_dir / 'berlin52.tsp'}")


if __name__ == "__main__":
    main()
