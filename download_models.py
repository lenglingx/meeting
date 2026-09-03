from pathlib import Path
from modelscope import snapshot_download

# 1. 导入配置和常量
# 注意：假设此文件在项目根目录，而 config 和 constants 在 app/ 目录下。
# 如果目录结构不同，请调整这里的 import 路径（例如 from config import settings）
from app.config import settings
from app.constants import MODELS


def download_model(local_name: str, model_id: str, base_dir: Path) -> None:
    """
    下载单个模型到指定的基础目录下，以 local_name 作为子文件夹名。
    """
    # 在配置的缓存目录下，为每个模型创建独立的子文件夹
    target_dir = base_dir / local_name

    print("=" * 70)
    print(f"正在下载：{model_id}")
    print(f"保存目录：{target_dir}")

    target_dir.mkdir(parents=True, exist_ok=True)

    # 执行下载
    downloaded_path = snapshot_download(
        model_id=model_id,
        local_dir=str(target_dir),
    )

    print(f"✅ 下载完成：{downloaded_path}")


if __name__ == "__main__":
    # 2. 从 settings 获取缓存目录，并解析为绝对路径
    base_cache_dir = Path(settings.MODELS_CACHE_DIR).resolve()
    base_cache_dir.mkdir(parents=True, exist_ok=True)
    
    print(f"📂 模型统一缓存基础目录: {base_cache_dir}")
    print(f"🖥️  当前配置的运行设备: {settings.DEVICE}")

    # 3. 从 constants 获取模型列表并遍历下载
    for local_name, model_id in MODELS.items():
        download_model(
            local_name=local_name,
            model_id=model_id,
            base_dir=base_cache_dir,
        )

    print("=" * 70)
    print("✅ 所有模型下载完成")