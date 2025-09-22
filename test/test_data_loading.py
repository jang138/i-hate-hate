"""
HuggingFace 데이터셋 로딩 함수 테스트
"""

import pandas as pd
from datasets import load_dataset


def load_data_from_hf(dataset_name, split, revision="main"):
    """HuggingFace에서 데이터셋 로드 → pandas DataFrame 반환"""
    print(f"=== {split} 데이터 로딩 시작 (revision: {revision}) ===")

    try:
        # HF Dataset 로드
        hf_dataset = load_dataset(dataset_name, split=split, revision=revision)
        print(f"HF Dataset 로드 성공: {len(hf_dataset)}개 샘플")

        # pandas DataFrame으로 변환
        dataset = hf_dataset.to_pandas()
        print(f"pandas DataFrame 변환 성공: {dataset.shape}")

        print("dataframe 의 형태")
        print("-" * 100)
        print(dataset.head())
        print("-" * 100)
        print("컬럼 정보:")
        print(dataset.info())
        print("-" * 100)
        print("output 값 분포:")
        if "output" in dataset.columns:
            print(dataset["output"].value_counts())

        return dataset

    except Exception as e:
        print(f"에러 발생: {e}")
        return None


def test_hf_dataset_loading(revision="main"):
    """HuggingFace 데이터셋 로딩 테스트"""
    dataset_name = "onestone11/nikl-hate-speech"

    print(f"HuggingFace 데이터셋 로딩 테스트 시작 (revision: {revision})")
    print("=" * 80)

    # 각 split 테스트
    splits = ["train", "validation", "test"]
    results = {}

    for split in splits:
        print(f"\n{split.upper()} 데이터 테스트")
        df = load_data_from_hf(dataset_name, split, revision)

        if df is not None:
            results[split] = {
                "success": True,
                "shape": df.shape,
                "columns": list(df.columns),
                "sample_data": df.head(2).to_dict(),
            }
            print(f"[SUCCESS] {split} 로딩 성공!")
        else:
            results[split] = {"success": False}
            print(f"[ERROR] {split} 로딩 실패!")

    # 결과 요약
    print("\n" + "=" * 80)
    print("테스트 결과 요약")
    print("=" * 80)

    for split, result in results.items():
        if result["success"]:
            print(
                f"[OK] {split:12s}: {result['shape'][0]:5d} rows × {result['shape'][1]} cols"
            )
        else:
            print(f"[FAIL] {split:12s}: 실패")

    # 데이터 일관성 체크
    if all(r["success"] for r in results.values()):
        print("\n데이터 일관성 체크")
        train_cols = set(results["train"]["columns"])
        val_cols = set(results["validation"]["columns"])
        test_cols = set(results["test"]["columns"])

        if train_cols == val_cols == test_cols:
            print("[OK] 모든 split의 컬럼이 일치합니다")
        else:
            print("[WARNING] 컬럼 불일치 발견:")
            print(f"   train: {train_cols}")
            print(f"   val:   {val_cols}")
            print(f"   test:  {test_cols}")

    return results


if __name__ == "__main__":
    # 테스트 실행
    # test_results = test_hf_dataset_loading()

    # 특정 버전 테스트
    test_results = test_hf_dataset_loading("v1.0")

    print("\n테스트 완료!")
    print("=" * 80)
