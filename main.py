"""
Project helper entrypoint.

This file intentionally keeps only high-level guidance so teammates can quickly
find the canonical runnable scripts after the project refactor.
"""


def main():
    print("KG-SAKT project entry guide")
    print("1) Data cleaning + KG build:")
    print(r"   .\.venv\Scripts\python.exe preprocess\clean_data.py")
    print("2) Train and evaluate:")
    print(r"   .\.venv\Scripts\python.exe utils\train_and_eval.py")
    print("3) Recommendation simulation:")
    print(r"   .\.venv\Scripts\python.exe utils\inference_recommend.py")
    print("4) Case visualization:")
    print(r"   .\.venv\Scripts\python.exe utils\case_study_viz.py")


if __name__ == "__main__":
    main()
