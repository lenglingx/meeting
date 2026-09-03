"""
Redis 连接层。
用途：
1. 缓存声纹 embedding，加速离线/实时说话人比对；
2. 实时会议 WebSocket 会话状态（当前说话人、部分转写文本）；
3. 多 worker 部署时的 pub/sub 广播（实时字幕推送）；
4. 离线会议分析任务的进度/状态标记。
"""

import logging

import redis.asyncio as redis

from app.config import settings

logger = logging.getLogger("redis_client")

# 全局连接池，所有 Redis 客户端共享，避免每次请求都新建 TCP 连接
redis_pool = redis.ConnectionPool.from_url(
    settings.REDIS_URL,
    decode_responses=True,   # 自动把 bytes 解码成 str，业务层不用手动 decode
    max_connections=50,
)


def get_redis() -> redis.Redis:
    """获取一个 Redis 客户端实例（基于共享连接池，轻量，可随时创建）"""
    return redis.Redis(connection_pool=redis_pool)


async def get_redis_dependency():
    """FastAPI 依赖注入版本"""
    client = get_redis()
    try:
        yield client
    finally:
        await client.close()


async def check_redis_connection() -> bool:
    """健康检查用：探测 Redis 是否可连接"""
    try:
        client = get_redis()
        pong = await client.ping()
        await client.close()
        return bool(pong)
    except Exception as e:  # noqa: BLE001
        logger.warning(f"Redis 连接检查失败: {e}")
        return False


async def close_redis_pool() -> None:
    """应用关闭时释放连接池，在 main.py 的 lifespan 里调用"""
    await redis_pool.disconnect()
