from transformers import AutoModelForSequenceClassification, AutoTokenizer
import os


def upload_model(model_dir, repo_name, revision="v1.0", commit_message=None):

    if not os.path.isabs(model_dir):
        project_root = os.path.dirname(os.path.dirname(__file__))
        model_dir = os.path.join(project_root, model_dir)

    print(f"Loading model from: {model_dir}")

    model = AutoModelForSequenceClassification.from_pretrained(model_dir)
    tokenizer = AutoTokenizer.from_pretrained("klue/bert-base")

    # 기본 커밋 메시지
    if commit_message is None:
        commit_message = f"Upload model and tokenizer - {revision}"

    # 모델 업로드
    model.push_to_hub(
        repo_name,
        commit_message=commit_message,
        revision=revision,
    )

    # 토크나이저 업로드
    tokenizer.push_to_hub(
        repo_name,
        commit_message=commit_message,
        revision=revision,
    )
    print(f"Upload completed: {repo_name} (revision: {revision})")


if __name__ == "__main__":
    project_root = os.path.dirname(os.path.dirname(__file__))
    model_dir = os.path.join(project_root, "best_model")

    upload_model(
        model_dir=model_dir,
        repo_name="team-sbai/bert-base-aeda-default",
        revision="v1.1-aeda",
        commit_message="AEDA augmentation with default params",
    )
