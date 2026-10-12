from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split


INFECTION_DRG_CODES = {
    "121", "122", "094", "095", "096", "757", "758", "759", "853", "854",
    "855", "689", "690", "485", "486", "487", "371", "372", "373", "097",
    "098", "099", "867", "868", "869", "856", "857", "858", "862", "863",
    "177", "178", "179", "548", "549", "550", "870", "871", "872", "865",
    "866", "969", "970", "974", "975", "976", "977",
}
MCC_DRG_CODES = {
    "288", "834", "280", "283", "559", "896", "915", "239", "255", "474",
    "616", "347", "268", "338", "341", "302", "518", "094", "725", "461",
    "408", "477", "553", "296", "308", "306", "222", "224", "226", "216",
    "219", "034", "602", "783", "786", "837", "846", "411", "414", "417",
    "190", "286", "432", "453", "380", "919", "088", "545", "231", "233",
    "235", "073", "023", "025", "056", "157", "637", "374", "441", "438",
    "444", "146", "643", "266", "319", "150", "391", "981", "037", "503",
    "562", "533", "535", "377", "388", "102", "291", "420", "423", "350",
    "353", "480", "969", "974", "304", "077", "757", "853", "727", "385",
    "196", "064", "020", "061", "656", "659", "689", "686", "695", "485",
    "495", "492", "820", "823", "840", "653", "163", "183", "368", "371",
    "808", "469", "595", "329", "435", "754", "722", "597", "551", "662",
    "606", "344", "640", "058", "826", "794", "054", "097", "987", "070",
    "067", "080", "939", "619", "539", "817", "831", "228", "314", "393",
    "356", "091", "124", "154", "628", "867", "922", "698", "673", "270",
    "963", "564", "515", "843", "907", "957", "205", "166", "579", "252",
    "152", "405", "542", "246", "248", "250", "273", "299", "040", "335",
    "258", "260", "242", "186", "199", "917", "856", "862", "791", "665",
    "175", "332", "811", "682", "177", "180", "814", "466", "100", "548",
    "871", "510", "947", "555", "193", "570", "573", "576", "592", "500",
    "456", "459", "471", "028", "799", "802", "326", "557", "625", "011",
    "001", "005", "668", "604", "913", "082", "085", "383", "693", "736",
    "739", "796", "805", "031", "865", "463", "622", "901", "140", "143",
    "210", "237", "276", "278", "321", "323", "359", "361", "397", "403",
    "426", "429", "447", "450", "521", "523", "650", "731", "984",
}


def build_features(source: pd.DataFrame, assets_dir: Path) -> pd.DataFrame:
    data = source.copy()
    data["length_of_stay"] = (
        data["length_of_stay"].astype(str).str.replace("+", "", regex=False).str.strip().astype(int)
    )
    data = data.rename(columns={"apr_mdc_code": "MDC"})
    columns = [
        "facility_name", "age_group", "gender", "ethnicity", "type_of_admission",
        "ccs_diagnosis_code", "ccs_procedure_code", "apr_drg_code", "MDC",
        "apr_severity_of_illness_code", "apr_risk_of_mortality",
        "emergency_department_indicator", "length_of_stay",
    ]
    master = data[columns]

    dx = pd.read_csv(assets_dir / "drg_codes.csv")
    dx["DRG_Medical"] = (dx["MS"] == "M").astype(int)
    dx["DRG_Surgical"] = (dx["MS"] == "P").astype(int)
    dx["DRG_Infection"] = dx["DRG"].astype(str).isin(INFECTION_DRG_CODES).astype(int)
    dx["DRG_MCC"] = dx["DRG"].astype(str).isin(MCC_DRG_CODES).astype(int)
    for code in range(1, 11):
        dx[f"MDC_{code}"] = (dx["MDC"] == code).astype(int)
    dx = dx.drop(columns=["MDC", "MS", "Description"])
    master = master.merge(dx, left_on="ccs_diagnosis_code", right_on="DRG", how="inner")

    px = pd.read_csv(assets_dir / "ccs_multi_pr_tool_2015.csv")
    procedure_map = {
        "Miscellaneous diagnostic and therapeutic procedures": "pocedure_others",
        "Operations on the musculoskeletal system": "pocedure_physical",
        "Operations on the cardiovascular system": "pocedure_cardiovascular",
        "Operations on the eye": "pocedure_eye",
        "Operations on the nose; mouth; and pharynx": "pocedure_sensory",
        "Operations on the nervous system": "pocedure_nervous",
        "Operations on the endocrine system": "pocedure_endocrine",
        "Operations on the ear": "pocedure_ear",
    }
    px["procedure_type"] = px["'CCS LVL 1 LABEL'"].map(procedure_map).fillna("pocedure_others")
    px["'ICD-9-CM CODE'"] = px["'ICD-9-CM CODE'"].astype(str).str.strip("'").astype(int)
    px = px[["'ICD-9-CM CODE'", "procedure_type"]].rename(columns={"'ICD-9-CM CODE'": "px_code"})
    master = master.merge(px, left_on="ccs_procedure_code", right_on="px_code", how="inner")

    weights = pd.read_excel(assets_dir / "2018-07-01_final_weights.xlsx", header=2)
    weights = weights[["APR-DRG", "Severity", "Service Intensity Weight "]].rename(
        columns={"APR-DRG": "apr_code", "Service Intensity Weight ": "service_intensity_weight", "Severity": "severity"}
    )
    master = master.merge(weights, left_on="apr_drg_code", right_on="apr_code", how="inner")
    master["facility_name"] = master["facility_name"].map({
        "Montefiore Medical Center - Henry & Lucy Moses Div": "1346294717",
        "Montefiore Medical Center - Wakefield Hospital": "1336125020",
        "Jacobi Medical Center": "1679587679",
        "Lincoln Medical & Mental Health Center": "1427063270",
        "SBH Health System": "1326148669",
        "Bronx-Lebanon Hospital Center - Fulton Division": "1245390855",
    })
    master["MDC"] = master["MDC"].map(
        {19: "Mental", 22: "Burn", 23: "Rehab", 20: "Drug/Alcohol Abuse", 25: "Trauma", 17: "_17"}
    ).fillna("Others")
    master = master.drop(columns=["ccs_diagnosis_code", "ccs_procedure_code", "apr_drg_code", "DRG", "px_code", "apr_code"])

    categorical = [
        "facility_name", "age_group", "gender", "ethnicity", "type_of_admission",
        "apr_severity_of_illness_code", "apr_risk_of_mortality",
        "emergency_department_indicator", "procedure_type", "severity", "MDC",
    ]
    features = pd.get_dummies(master.drop(columns=["length_of_stay"]), columns=categorical)
    features["service_intensity_weight"] = np.log1p(features["service_intensity_weight"])
    features = features.drop(
        columns=["DRG_Infection", "DRG_MCC", "MDC_6", "MDC_7", "MDC_8", "MDC_9", "MDC_10", "MDC_Others"],
        errors="ignore",
    )
    features = features.replace([np.inf, -np.inf], np.nan).fillna(0)
    return pd.concat([features, master[["length_of_stay"]]], axis=1)


def main() -> None:
    parser = argparse.ArgumentParser(description="Prepare LOS train/validation/test data.")
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--assets-dir", type=Path, default=Path("data"))
    args = parser.parse_args()
    frame = build_features(pd.read_csv(args.input), args.assets_dir)
    x = frame.drop(columns=["length_of_stay"])
    y = frame[["length_of_stay"]]
    x_train, x_holdout, y_train, y_holdout = train_test_split(x, y, test_size=0.4, random_state=42)
    x_valid, x_test, y_valid, y_test = train_test_split(
        x_holdout, y_holdout, test_size=0.5, random_state=42
    )
    args.output_dir.mkdir(parents=True, exist_ok=True)
    for name, value in {
        "x_train": x_train, "y_train": y_train,
        "x_valid": x_valid, "y_valid": y_valid,
        "x_test": x_test, "y_test": y_test,
    }.items():
        value.to_csv(args.output_dir / f"{name}.csv", index=False)
    (args.output_dir / "feature_columns.json").write_text(json.dumps(list(x.columns), indent=2) + "\n")
    print(f"Prepared {len(frame)} rows with {len(x.columns)} features in {args.output_dir}")


if __name__ == "__main__":
    main()
